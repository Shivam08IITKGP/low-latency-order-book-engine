#include "order_book.h"
#include "messages.h"
#include "timing.h"
#include <iostream>
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

// Find the lowest ask price ≥ start_price with quantity > 0.
uint64_t OrderBook::find_next_ask(uint64_t start_price)
{
    for (uint64_t price = start_price; price <= MAX_PRICE; price++)
    {
        size_t   idx    = price / 64;
        size_t   bit    = price % 64;
        uint64_t masked = ask_bitmap[idx] & (~0ULL << bit);
        if (masked)
            return idx * 64 + __builtin_ctzll(masked);

        // No bit in this block — scan remaining blocks
        for (size_t i = idx + 1; i < BITMAP_SIZE; i++)
        {
            if (ask_bitmap[i])
                return i * 64 + __builtin_ctzll(ask_bitmap[i]);
        }
        return MAX_PRICE + 1; // Book is empty on ask side
    }
    return MAX_PRICE + 1;
}

// Find the highest bid price ≤ start_price with quantity > 0.
uint64_t OrderBook::find_next_bid(uint64_t start_price)
{
    for (int64_t price = start_price; price >= 0; price--)
    {
        size_t   idx    = price / 64;
        size_t   bit    = price % 64;
        uint64_t masked = bid_bitmap[idx] & ((1ULL << (bit + 1)) - 1);
        if (masked)
            return idx * 64 + (63 - __builtin_clzll(masked));

        // No bit in this block — scan remaining blocks downward
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
// NEW ORDER ('N')
// ----------------------------------------------------------------

void OrderBook::addOrder(uint64_t id, char side, uint64_t price, uint32_t quantity)
{
    // 0='B', 1='S' encoding for the 1-bit side field
    order_lookup[id] = { price, quantity, static_cast<uint64_t>(side == 'B' ? 0 : 1) };

    if (side == 'B')
    {
        bids[price] += quantity;
        set_bid_level(price);
        max_bid_price = std::max(price, max_bid_price);
    }
    else
    {
        asks[price] += quantity;
        set_ask_level(price);
        min_ask_price = std::min(price, min_ask_price);
    }

    UpdateMessage msg{ id, price, get_timestamp_ns(), quantity, 'N', side };
    updateBuffer.push(msg);
}

// ----------------------------------------------------------------
// CANCEL ORDER ('X')
// ----------------------------------------------------------------

void OrderBook::cancelOrder(uint64_t id)
{
    if (id >= order_lookup.size())
        return;

    OrderInfo& info = order_lookup[id];
    if (info.quantity == 0)
        return;

    if (info.side == 0) // Buy
    {
        bids[info.price] -= info.quantity;
        if (bids[info.price] == 0)
        {
            clear_bid_level(info.price);
            if (info.price == max_bid_price)
                max_bid_price = find_next_bid(max_bid_price - 1);
        }
    }
    else // Sell
    {
        asks[info.price] -= info.quantity;
        if (asks[info.price] == 0)
        {
            clear_ask_level(info.price);
            if (info.price == min_ask_price)
                min_ask_price = find_next_ask(min_ask_price + 1);
        }
    }

    char side_char = (info.side == 0) ? 'B' : 'S';
    UpdateMessage msg{ id, info.price, get_timestamp_ns(),
                       static_cast<uint32_t>(info.quantity), 'X', side_char };
    updateBuffer.push(msg);

    info.quantity = 0; // Mark as removed
}

// ----------------------------------------------------------------
// TRADE ('T') — reduces quantity on both matched sides
// ----------------------------------------------------------------

void OrderBook::executeTrade(uint64_t buy_id, uint64_t sell_id, uint32_t qty)
{
    if (order_lookup[buy_id].quantity == 0 || order_lookup[sell_id].quantity == 0)
        return;

    // Buy side
    OrderInfo& buy = order_lookup[buy_id];
    bids[buy.price] -= qty;
    buy.quantity    -= qty;
    if (bids[buy.price] == 0)
    {
        clear_bid_level(buy.price);
        if (buy.price == max_bid_price)
            max_bid_price = find_next_bid(max_bid_price - 1);
    }

    // Sell side
    OrderInfo& sell = order_lookup[sell_id];
    asks[sell.price] -= qty;
    sell.quantity    -= qty;
    if (asks[sell.price] == 0)
    {
        clear_ask_level(sell.price);
        if (sell.price == min_ask_price)
            min_ask_price = find_next_ask(min_ask_price + 1);
    }

    total_traded_volume += qty;

    UpdateMessage msg{ buy_id, buy.price, get_timestamp_ns(), qty, 'T', 'B' };
    updateBuffer.push(msg);
}

// ----------------------------------------------------------------
// MODIFY ORDER ('M') — smart priority preservation
// ----------------------------------------------------------------
// Priority rules (exchange-standard):
//   Price / side changed  → cancel + re-add (lose priority)
//   Quantity increased    → cancel + re-add (lose priority)
//   Quantity decreased    → update in-place (KEEP priority)
//   Quantity unchanged    → no-op

void OrderBook::modifyOrder(uint64_t id, char new_side,
                             uint64_t new_price, uint32_t new_quantity)
{
    if (id >= order_lookup.size())
        return;

    OrderInfo& info = order_lookup[id];
    if (info.quantity == 0)
        return;

    uint64_t new_side_bit = (new_side == 'B' ? 0 : 1);

    // Case 1 & 2: Lose priority
    if (new_price != info.price || new_side_bit != info.side ||
        new_quantity > static_cast<uint32_t>(info.quantity))
    {
        cancelOrder(id);
        addOrder(id, new_side, new_price, new_quantity);
        return;
    }

    // Case 3: Quantity decreased — update in-place, maintain queue position
    if (new_quantity < static_cast<uint32_t>(info.quantity))
    {
        uint32_t decrease = static_cast<uint32_t>(info.quantity) - new_quantity;

        if (info.side == 0)
        {
            bids[info.price] -= decrease;
            if (bids[info.price] == 0)
            {
                clear_bid_level(info.price);
                if (info.price == max_bid_price)
                    max_bid_price = find_next_bid(max_bid_price - 1);
            }
        }
        else
        {
            asks[info.price] -= decrease;
            if (asks[info.price] == 0)
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

    // Case 4: Quantity unchanged — no-op
}

// ----------------------------------------------------------------
// MARKET ORDER — walks the book, bitmap-optimised
// ----------------------------------------------------------------

void OrderBook::executeMarketOrder(char side, uint32_t quantity)
{
    if (side == 'B')
    {
        uint32_t remaining = quantity;

        for (uint64_t price = min_ask_price; price <= MAX_PRICE && remaining > 0; price++)
        {
            if (asks[price] == 0)
                continue;

            uint32_t matched = std::min(remaining, asks[price]);
            asks[price]         -= matched;
            remaining           -= matched;
            total_traded_volume += matched;

            UpdateMessage msg{ 0, price, get_timestamp_ns(), matched, 'T', 'B' };
            updateBuffer.push(msg);

            if (asks[price] == 0)
            {
                clear_ask_level(price);
                if (price == min_ask_price)
                    min_ask_price = find_next_ask(min_ask_price + 1);
            }
        }

        if (remaining > 0)
            std::cout << "[WARNING] Market Buy partially filled. Unfilled: " << remaining << "\n";
    }
    else
    {
        uint32_t remaining = quantity;

        for (uint64_t price = max_bid_price; price > 0 && remaining > 0; price--)
        {
            if (bids[price] == 0)
                continue;

            uint32_t matched = std::min(remaining, bids[price]);
            bids[price]         -= matched;
            remaining           -= matched;
            total_traded_volume += matched;

            UpdateMessage msg{ 0, price, get_timestamp_ns(), matched, 'T', 'S' };
            updateBuffer.push(msg);

            if (bids[price] == 0)
            {
                clear_bid_level(price);
                if (price == max_bid_price)
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
              << (max_bid_price > 0 ? bids[max_bid_price] : 0)
              << " @ " << max_bid_price
              << " | Ask: "
              << (min_ask_price <= MAX_PRICE ? asks[min_ask_price] : 0)
              << " @ " << min_ask_price << "\n";
}

// ----------------------------------------------------------------
// Memory pre-faulting
// ----------------------------------------------------------------

void OrderBook::pre_fault_memory()
{
    const size_t page_size = 4096;

    for (size_t i = 0; i < order_lookup.size(); i += page_size / sizeof(OrderInfo))
        order_lookup[i].quantity = 0;

    for (size_t i = 0; i < bids.size(); i += page_size / sizeof(uint32_t))
    {
        bids[i] = 0;
        asks[i] = 0;
    }

    for (size_t i = 0; i < bid_bitmap.size(); i += page_size / sizeof(uint64_t))
    {
        bid_bitmap[i] = 0;
        ask_bitmap[i] = 0;
    }
}