#include "order_book.h"
#include "messages.h"
#include "timing.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

// ----------------------------------------------------------------
// Bitmap helpers
// ----------------------------------------------------------------

inline void OrderBook::set_bid_level(uint64_t price)
{
    bid_bitmap[price / 64] |= (1ULL << (price % 64));
}

inline void OrderBook::clear_bid_level(uint64_t price)
{
    bid_bitmap[price / 64] &= ~(1ULL << (price % 64));
}

inline void OrderBook::set_ask_level(uint64_t price)
{
    ask_bitmap[price / 64] |= (1ULL << (price % 64));
}

inline void OrderBook::clear_ask_level(uint64_t price)
{
    ask_bitmap[price / 64] &= ~(1ULL << (price % 64));
}

// ----------------------------------------------------------------
// Best-price search using bit-scanning intrinsics
// ----------------------------------------------------------------

// Find the lowest ask price >= start_price with quantity > 0.
uint64_t OrderBook::find_next_ask(uint64_t start_price)
{
    for (uint64_t price = start_price; price <= MAX_PRICE; price++)
    {
        size_t   idx    = price / 64;
        size_t   bit    = price % 64;
        uint64_t masked = ask_bitmap[idx] & (~0ULL << bit);
        if (masked)
            return idx * 64 + __builtin_ctzll(masked);

        // No bit in this block -- scan remaining blocks
        for (size_t i = idx + 1; i < BITMAP_SIZE; i++)
        {
            if (ask_bitmap[i])
                return i * 64 + __builtin_ctzll(ask_bitmap[i]);
        }
        return MAX_PRICE + 1; // Book is empty on ask side
    }
    return MAX_PRICE + 1;
}

// Find the highest bid price <= start_price with quantity > 0.
uint64_t OrderBook::find_next_bid(uint64_t start_price)
{
    for (int64_t price = start_price; price >= 0; price--)
    {
        size_t   idx    = price / 64;
        size_t   bit    = price % 64;
        uint64_t masked = bid_bitmap[idx] & ((1ULL << (bit + 1)) - 1);
        if (masked)
            return idx * 64 + (63 - __builtin_clzll(masked));

        // No bit in this block -- scan remaining blocks downward
        for (int64_t i = idx - 1; i >= 0; i--)
        {
            if (bid_bitmap[i])
                return i * 64 + (63 - __builtin_clzll(bid_bitmap[i]));
        }
        return 0; // Book is empty on bid side
    }
    return 0;
}

// ----------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------

OrderBook::OrderBook()
{
    // std::array members are zero-initialised via {} in the class definition.
    // total_traded_volume is also initialised there.
}

// ----------------------------------------------------------------
// NEW ORDER ('N')  -- Price-Time (FIFO) Priority Matching Engine
// ----------------------------------------------------------------
// Matching rule: incoming order walks the opposite side's FIFO queue
// from the best price level outward. Within each price level, the
// oldest resting order is always filled first (time priority).
//
// Lazy cancellation: cancelled orders stay in the intrusive linked
// list with quantity == 0. They are silently skipped here in O(1)
// per node, avoiding any list-traversal in cancelOrder.

void OrderBook::addOrder(uint64_t id, char side, uint64_t price, uint32_t quantity,
                         uint32_t participant_id /*= 0*/, OrderType type /*= OrderType::GTC*/)
{
    uint32_t remaining_qty = quantity;

    if (side == 'B')
    {
        // ---- Match against resting Asks, best price first ----
        while (remaining_qty > 0 && min_ask_price <= MAX_PRICE && price >= min_ask_price)
        {
            PriceLevel& level = ask_levels[min_ask_price];

            // Flush lazily-cancelled (qty==0) nodes from the head of the queue.
            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead   = level.head;
                level.head      = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_ask_level(min_ask_price);
                min_ask_price = find_next_ask(min_ask_price + 1);
                continue;
            }

            // Match against the front of the queue (highest time-priority)
            OrderInfo& resting  = order_lookup[level.head];

            // ---- Self-Trade Prevention (STP) ----
            if (participant_id != 0 && resting.participant_id == participant_id)
            {
                // Action: Cancel the RESTING order (Cancel-Resting)
                // In HFT, we frequently cancel the resting order to allow the aggressor to find liquidity.
                cancelOrder(level.head);
                continue; // Re-evaluate this price level
            }

            uint32_t matched  = std::min(remaining_qty, resting.quantity);

            resting.quantity    -= matched;
            remaining_qty       -= matched;
            level.total_qty     -= matched;
            total_traded_volume   += matched;
            traded_notional_value += (static_cast<uint64_t>(matched) * min_ask_price);

            UpdateMessage trade_msg{ level.head, min_ask_price,
                                     get_timestamp_ns(), matched, 'T', 'B' };
            updateBuffer.push(trade_msg);

            if (resting.quantity == 0)
            {
                // Dequeue the fully-filled order
                uint32_t filled_id     = level.head;
                level.head             = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[filled_id].next_id = NULL_ORDER;
            }

            if (level.total_qty == 0)
            {
                clear_ask_level(min_ask_price);
                min_ask_price = find_next_ask(min_ask_price + 1);
            }
        }
    }
    else // side == 'S'
    {
        // ---- Match against resting Bids, best price first ----
        while (remaining_qty > 0 && max_bid_price > 0 && price <= max_bid_price)
        {
            PriceLevel& level = bid_levels[max_bid_price];

            // Flush lazily-cancelled nodes from the head of the queue.
            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead   = level.head;
                level.head      = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_bid_level(max_bid_price);
                max_bid_price = find_next_bid(max_bid_price - 1);
                continue;
            }

            // Match against the front of the queue (highest time-priority)
            OrderInfo& resting  = order_lookup[level.head];

            // ---- Self-Trade Prevention (STP) ----
            if (participant_id != 0 && resting.participant_id == participant_id)
            {
                cancelOrder(level.head);
                continue;
            }

            uint32_t matched  = std::min(remaining_qty, resting.quantity);

            resting.quantity    -= matched;
            remaining_qty       -= matched;
            level.total_qty     -= matched;
            total_traded_volume   += matched;
            traded_notional_value += (static_cast<uint64_t>(matched) * max_bid_price);

            UpdateMessage trade_msg{ level.head, max_bid_price,
                                     get_timestamp_ns(), matched, 'T', 'S' };
            updateBuffer.push(trade_msg);

            if (resting.quantity == 0)
            {
                uint32_t filled_id     = level.head;
                level.head             = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[filled_id].next_id = NULL_ORDER;
            }

            if (level.total_qty == 0)
            {
                clear_bid_level(max_bid_price);
                max_bid_price = find_next_bid(max_bid_price - 1);
            }
        }
    }

    // IOC: discard any unfilled remainder immediately -- do NOT post to book
    if (type == OrderType::IOC)
    {
        if (remaining_qty > 0)
        {
            UpdateMessage cancel_msg{ id, price, get_timestamp_ns(), remaining_qty, 'X', side };
            updateBuffer.push(cancel_msg);
        }
        return;
    }

    // FOK: if the ENTIRE quantity wasn't filled, cancel the whole order
    if (type == OrderType::FOK)
    {
        if (remaining_qty > 0)
        {
            UpdateMessage cancel_msg{ id, price, get_timestamp_ns(), quantity, 'X', side };
            updateBuffer.push(cancel_msg);
        }
        return;
    }

    // GTC (default): append remaining qty to the BACK of the price level's
    // FIFO queue -- this preserves time priority for all existing resting orders.
    if (remaining_qty > 0)
    {
        order_lookup[id] = { static_cast<uint32_t>(price), remaining_qty,
                             NULL_ORDER, participant_id,
                             static_cast<uint8_t>(side == 'B' ? 0 : 1), {} };

        if (side == 'B')
        {
            PriceLevel& level = bid_levels[price];
            if (level.tail == NULL_ORDER)
                level.head = level.tail = static_cast<uint32_t>(id);
            else
            {
                order_lookup[level.tail].next_id = static_cast<uint32_t>(id);
                level.tail = static_cast<uint32_t>(id);
            }
            level.total_qty += remaining_qty;
            set_bid_level(price);
            max_bid_price = std::max(price, max_bid_price);
        }
        else
        {
            PriceLevel& level = ask_levels[price];
            if (level.tail == NULL_ORDER)
                level.head = level.tail = static_cast<uint32_t>(id);
            else
            {
                order_lookup[level.tail].next_id = static_cast<uint32_t>(id);
                level.tail = static_cast<uint32_t>(id);
            }
            level.total_qty += remaining_qty;
            set_ask_level(price);
            min_ask_price = std::min(price, min_ask_price);
        }

        UpdateMessage post_msg{ id, price, get_timestamp_ns(), remaining_qty, 'N', side };
        updateBuffer.push(post_msg);
    }
}

// ----------------------------------------------------------------
// CANCEL ORDER ('X')
// ----------------------------------------------------------------
// O(1) lazy cancellation: subtract qty from the level's total_qty
// immediately (keeping the book's visible depth correct), then zero
// out info.quantity so the matching loop silently skips this node.
// The node is not physically unlinked here -- that happens the next
// time the head of the queue is visited during matching.

void OrderBook::cancelOrder(uint64_t id)
{
    if (id >= order_lookup.size())
        return;

    OrderInfo& info = order_lookup[id];
    if (info.quantity == 0)
        return;

    if (info.side == 0) // Buy
    {
        PriceLevel& level = bid_levels[info.price];
        level.total_qty  -= info.quantity;
        if (level.total_qty == 0)
        {
            clear_bid_level(info.price);
            if (info.price == max_bid_price)
                max_bid_price = find_next_bid(max_bid_price - 1);
        }
    }
    else // Sell
    {
        PriceLevel& level = ask_levels[info.price];
        level.total_qty  -= info.quantity;
        if (level.total_qty == 0)
        {
            clear_ask_level(info.price);
            if (info.price == min_ask_price)
                min_ask_price = find_next_ask(min_ask_price + 1);
        }
    }

    char side_char = (info.side == 0) ? 'B' : 'S';
    UpdateMessage msg{ id, info.price, get_timestamp_ns(),
                       info.quantity, 'X', side_char };
    updateBuffer.push(msg);

    info.quantity = 0; // Lazy: node stays linked, skipped during matching
}

// ----------------------------------------------------------------
// TRADE ('T') -- reduces quantity on both matched sides
// ----------------------------------------------------------------

void OrderBook::executeTrade(uint64_t buy_id, uint64_t sell_id, uint32_t qty)
{
    if (order_lookup[buy_id].quantity == 0 || order_lookup[sell_id].quantity == 0)
        return;

    // Buy side
    OrderInfo& buy    = order_lookup[buy_id];
    PriceLevel& blvl  = bid_levels[buy.price];
    buy.quantity     -= qty;
    blvl.total_qty   -= qty;
    if (blvl.total_qty == 0)
    {
        clear_bid_level(buy.price);
        if (buy.price == max_bid_price)
            max_bid_price = find_next_bid(max_bid_price - 1);
    }

    // Sell side
    OrderInfo& sell   = order_lookup[sell_id];
    PriceLevel& alvl  = ask_levels[sell.price];
    sell.quantity    -= qty;
    alvl.total_qty   -= qty;
    if (alvl.total_qty == 0)
    {
        clear_ask_level(sell.price);
        if (sell.price == min_ask_price)
            min_ask_price = find_next_ask(min_ask_price + 1);
    }

    total_traded_volume   += qty;
    traded_notional_value += (static_cast<uint64_t>(qty) * buy.price);

    UpdateMessage msg{ buy_id, buy.price, get_timestamp_ns(), qty, 'T', 'B' };
    updateBuffer.push(msg);
}

// ----------------------------------------------------------------
// MODIFY ORDER ('M') -- smart priority preservation
// ----------------------------------------------------------------
// Priority rules (exchange-standard):
//   Price / side changed  -> cancel + re-add (lose priority)
//   Quantity increased    -> cancel + re-add (lose priority)
//   Quantity decreased    -> update in-place (KEEP priority)
//   Quantity unchanged    -> no-op

void OrderBook::modifyOrder(uint64_t id, char new_side,
                             uint64_t new_price, uint32_t new_quantity)
{
    if (id >= order_lookup.size())
        return;

    OrderInfo& info = order_lookup[id];
    if (info.quantity == 0)
        return;

    uint8_t new_side_bit = (new_side == 'B' ? 0 : 1);

    // Case 1 & 2: Price/side change or quantity increase -> lose priority
    if (new_price != info.price || new_side_bit != info.side ||
        new_quantity > info.quantity)
    {
        cancelOrder(id);
        addOrder(id, new_side, new_price, new_quantity);
        return;
    }

    // Case 3: Quantity decreased -- update in-place, KEEP queue position
    if (new_quantity < info.quantity)
    {
        uint32_t decrease = info.quantity - new_quantity;

        if (info.side == 0)
        {
            PriceLevel& level = bid_levels[info.price];
            level.total_qty  -= decrease;
            if (level.total_qty == 0)
            {
                clear_bid_level(info.price);
                if (info.price == max_bid_price)
                    max_bid_price = find_next_bid(max_bid_price - 1);
            }
        }
        else
        {
            PriceLevel& level = ask_levels[info.price];
            level.total_qty  -= decrease;
            if (level.total_qty == 0)
            {
                clear_ask_level(info.price);
                if (info.price == min_ask_price)
                    min_ask_price = find_next_ask(min_ask_price + 1);
            }
        }

        info.quantity = new_quantity;

        UpdateMessage msg{ id, new_price, get_timestamp_ns(), new_quantity, 'M', new_side };
        updateBuffer.push(msg);
    }

    // Case 4: Quantity unchanged -- no-op
}

// ----------------------------------------------------------------
// MARKET ORDER -- walks the book, bitmap-optimised
// ----------------------------------------------------------------

void OrderBook::executeMarketOrder(char side, uint32_t quantity)
{
    if (side == 'B')
    {
        uint32_t remaining = quantity;

        while (remaining > 0 && min_ask_price <= MAX_PRICE)
        {
            PriceLevel& level = ask_levels[min_ask_price];

            // Flush lazily-cancelled nodes from head
            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead   = level.head;
                level.head      = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_ask_level(min_ask_price);
                min_ask_price = find_next_ask(min_ask_price + 1);
                continue;
            }

            OrderInfo& resting  = order_lookup[level.head];
            uint32_t   matched  = std::min(remaining, resting.quantity);

            resting.quantity    -= matched;
            remaining           -= matched;
            level.total_qty     -= matched;
            total_traded_volume   += matched;
            traded_notional_value += (static_cast<uint64_t>(matched) * min_ask_price);

            UpdateMessage msg{ level.head, min_ask_price,
                               get_timestamp_ns(), matched, 'T', 'B' };
            updateBuffer.push(msg);

            if (resting.quantity == 0)
            {
                uint32_t filled_id = level.head;
                level.head         = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[filled_id].next_id = NULL_ORDER;
            }

            if (level.total_qty == 0)
            {
                clear_ask_level(min_ask_price);
                min_ask_price = find_next_ask(min_ask_price + 1);
            }
        }

        if (remaining > 0)
            std::cout << "[WARNING] Market Buy partially filled. Unfilled: " << remaining << "\n";
    }
    else
    {
        uint32_t remaining = quantity;

        while (remaining > 0 && max_bid_price > 0)
        {
            PriceLevel& level = bid_levels[max_bid_price];

            // Flush lazily-cancelled nodes from head
            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead   = level.head;
                level.head      = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_bid_level(max_bid_price);
                max_bid_price = find_next_bid(max_bid_price - 1);
                continue;
            }

            OrderInfo& resting  = order_lookup[level.head];
            uint32_t   matched  = std::min(remaining, resting.quantity);

            resting.quantity    -= matched;
            remaining           -= matched;
            level.total_qty     -= matched;
            total_traded_volume   += matched;
            traded_notional_value += (static_cast<uint64_t>(matched) * max_bid_price);

            UpdateMessage msg{ level.head, max_bid_price,
                               get_timestamp_ns(), matched, 'T', 'S' };
            updateBuffer.push(msg);

            if (resting.quantity == 0)
            {
                uint32_t filled_id = level.head;
                level.head         = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[filled_id].next_id = NULL_ORDER;
            }

            if (level.total_qty == 0)
            {
                clear_bid_level(max_bid_price);
                max_bid_price = find_next_bid(max_bid_price - 1);
            }
        }

        if (remaining > 0)
            std::cout << "[WARNING] Market Sell partially filled. Unfilled: " << remaining << "\n";
    }
}

// ----------------------------------------------------------------
// Diagnostics
// ----------------------------------------------------------------

void OrderBook::printTopOfBook() const
{
    std::cout << "   [BOOK] Bid: "
              << (max_bid_price > 0 ? bid_levels[max_bid_price].total_qty : 0)
              << " @ " << max_bid_price
              << " | Ask: "
              << (min_ask_price <= MAX_PRICE ? ask_levels[min_ask_price].total_qty : 0)
              << " @ " << min_ask_price 
              << " | VWAP: " << std::fixed << std::setprecision(2) << getVWAP() << "\n";
}

bool OrderBook::saveSnapshot(const std::string& filename) const
{
    std::ofstream os(filename, std::ios::binary);
    if (!os) return false;

    os.write(reinterpret_cast<const char*>(&max_bid_price), sizeof(max_bid_price));
    os.write(reinterpret_cast<const char*>(&min_ask_price), sizeof(min_ask_price));
    os.write(reinterpret_cast<const char*>(&total_traded_volume), sizeof(total_traded_volume));
    os.write(reinterpret_cast<const char*>(&traded_notional_value), sizeof(traded_notional_value));
    
    os.write(reinterpret_cast<const char*>(bid_levels.data()), bid_levels.size() * sizeof(PriceLevel));
    os.write(reinterpret_cast<const char*>(ask_levels.data()), ask_levels.size() * sizeof(PriceLevel));
    os.write(reinterpret_cast<const char*>(bid_bitmap.data()), bid_bitmap.size() * sizeof(uint64_t));
    os.write(reinterpret_cast<const char*>(ask_bitmap.data()), ask_bitmap.size() * sizeof(uint64_t));
    os.write(reinterpret_cast<const char*>(order_lookup.data()), order_lookup.size() * sizeof(OrderInfo));

    return os.good();
}

bool OrderBook::loadSnapshot(const std::string& filename)
{
    std::ifstream is(filename, std::ios::binary);
    if (!is) return false;

    is.read(reinterpret_cast<char*>(&max_bid_price), sizeof(max_bid_price));
    is.read(reinterpret_cast<char*>(&min_ask_price), sizeof(min_ask_price));
    is.read(reinterpret_cast<char*>(&total_traded_volume), sizeof(total_traded_volume));
    is.read(reinterpret_cast<char*>(&traded_notional_value), sizeof(traded_notional_value));

    is.read(reinterpret_cast<char*>(bid_levels.data()), bid_levels.size() * sizeof(PriceLevel));
    is.read(reinterpret_cast<char*>(ask_levels.data()), ask_levels.size() * sizeof(PriceLevel));
    is.read(reinterpret_cast<char*>(bid_bitmap.data()), bid_bitmap.size() * sizeof(uint64_t));
    is.read(reinterpret_cast<char*>(ask_bitmap.data()), ask_bitmap.size() * sizeof(uint64_t));
    is.read(reinterpret_cast<char*>(order_lookup.data()), order_lookup.size() * sizeof(OrderInfo));

    return is.good();
}

// ----------------------------------------------------------------
// Memory pre-faulting
// ----------------------------------------------------------------

void OrderBook::pre_fault_memory()
{
    const size_t page_size = 4096;

    // Touch one element per page to force OS to allocate all physical pages
    // upfront, eliminating page-fault latency spikes during the hot path.

    for (size_t i = 0; i < order_lookup.size(); i += page_size / sizeof(OrderInfo))
        order_lookup[i].quantity = 0;

    for (size_t i = 0; i < bid_levels.size(); i += page_size / sizeof(PriceLevel))
    {
        bid_levels[i].total_qty = 0;
        ask_levels[i].total_qty = 0;
    }

    for (size_t i = 0; i < bid_bitmap.size(); i += page_size / sizeof(uint64_t))
    {
        bid_bitmap[i] = 0;
        ask_bitmap[i] = 0;
    }
}