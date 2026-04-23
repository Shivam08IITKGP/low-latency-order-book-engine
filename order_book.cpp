#include "order_book.h"
#include "messages.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <iomanip>

/**
 * OrderBook Implementation
 * 
 * DESIGN PRINCIPLES:
 * 1. O(1) order lookup via direct-indexed array.
 * 2. O(1) best bid/ask search via bitmask + __builtin_ctzll.
 * 3. FIFO price-time priority via intrusive singly-linked lists.
 * 4. Zero heap allocation in the matching hot path.
 */

OrderBook::OrderBook()
{
    std::fill(bid_bitmap.begin(), bid_bitmap.end(), 0);
    std::fill(ask_bitmap.begin(), ask_bitmap.end(), 0);
}

void OrderBook::addOrder(uint64_t id, char side, uint64_t price, uint32_t quantity,
                         uint32_t participant_id, OrderType type)
{
    if (price > MAX_PRICE || price == 0) return;

    uint32_t remaining = quantity;

    if (side == 'B')
    {
        // Match against Asks
        while (remaining > 0 && min_ask_price <= MAX_PRICE && price >= min_ask_price)
        {
            PriceLevel& level = ask_levels[min_ask_price];

            // Lazy cancellation cleanup
            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead = level.head;
                level.head = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_ask_level(min_ask_price);
                min_ask_price = find_next_ask(min_ask_price + 1);
                continue;
            }

            OrderInfo& resting = order_lookup[level.head];

            // Self-Trade Prevention (STP)
            if (participant_id != 0 && resting.participant_id == participant_id)
            {
                uint32_t stp_id = level.head;
                level.total_qty -= resting.quantity;
                resting.quantity = 0;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[stp_id].next_id = NULL_ORDER;
                continue;
            }

            uint32_t matched = std::min(remaining, resting.quantity);
            resting.quantity -= matched;
            remaining -= matched;
            level.total_qty -= matched;
            total_traded_volume += matched;
            traded_notional_value += (static_cast<uint64_t>(matched) * min_ask_price);

            // Log Trade
            UpdateMessage trade_msg{ level.head, min_ask_price, get_timestamp_raw(), matched, 'T', 'B' };
            updateBuffer.push(trade_msg);

            if (resting.quantity == 0)
            {
                uint32_t filled_id = level.head;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[filled_id].next_id = NULL_ORDER;
            }

            if (level.total_qty == 0)
            {
                clear_ask_level(min_ask_price);
                min_ask_price = find_next_ask(min_ask_price + 1);
            }
        }

        if (remaining > 0 && type == OrderType::GTC)
        {
            OrderInfo& info = order_lookup[id];
            info.price = static_cast<uint32_t>(price);
            info.quantity = remaining;
            info.side = 0;
            info.participant_id = participant_id;
            info.next_id = NULL_ORDER;

            PriceLevel& level = bid_levels[price];
            if (level.head == NULL_ORDER)
            {
                level.head = level.tail = static_cast<uint32_t>(id);
                set_bid_level(price);
                max_bid_price = std::max(price, max_bid_price);
            }
            else
            {
                order_lookup[level.tail].next_id = static_cast<uint32_t>(id);
                level.tail = static_cast<uint32_t>(id);
            }
            level.total_qty += remaining;

            UpdateMessage post_msg{ id, price, get_timestamp_raw(), remaining, 'N', 'B' };
            updateBuffer.push(post_msg);
        }
    }
    else // side == 'S'
    {
        while (remaining > 0 && max_bid_price > 0 && price <= max_bid_price)
        {
            PriceLevel& level = bid_levels[max_bid_price];

            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead = level.head;
                level.head = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_bid_level(max_bid_price);
                max_bid_price = find_next_bid(max_bid_price - 1);
                continue;
            }

            OrderInfo& resting = order_lookup[level.head];

            if (participant_id != 0 && resting.participant_id == participant_id)
            {
                uint32_t stp_id = level.head;
                level.total_qty -= resting.quantity;
                resting.quantity = 0;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[stp_id].next_id = NULL_ORDER;
                continue;
            }

            uint32_t matched = std::min(remaining, resting.quantity);
            resting.quantity -= matched;
            remaining -= matched;
            level.total_qty -= matched;
            total_traded_volume += matched;
            traded_notional_value += (static_cast<uint64_t>(matched) * max_bid_price);

            UpdateMessage trade_msg{ level.head, max_bid_price, get_timestamp_raw(), matched, 'T', 'S' };
            updateBuffer.push(trade_msg);

            if (resting.quantity == 0)
            {
                uint32_t filled_id = level.head;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER) level.tail = NULL_ORDER;
                order_lookup[filled_id].next_id = NULL_ORDER;
            }

            if (level.total_qty == 0)
            {
                clear_bid_level(max_bid_price);
                max_bid_price = find_next_bid(max_bid_price - 1);
            }
        }

        if (remaining > 0 && type == OrderType::GTC)
        {
            OrderInfo& info = order_lookup[id];
            info.price = static_cast<uint32_t>(price);
            info.quantity = remaining;
            info.side = 1;
            info.participant_id = participant_id;
            info.next_id = NULL_ORDER;

            PriceLevel& level = ask_levels[price];
            if (level.head == NULL_ORDER)
            {
                level.head = level.tail = static_cast<uint32_t>(id);
                set_ask_level(price);
                min_ask_price = std::min(price, min_ask_price);
            }
            else
            {
                order_lookup[level.tail].next_id = static_cast<uint32_t>(id);
                level.tail = static_cast<uint32_t>(id);
            }
            level.total_qty += remaining;

            UpdateMessage post_msg{ id, price, get_timestamp_raw(), remaining, 'N', 'S' };
            updateBuffer.push(post_msg);
        }
    }
}

void OrderBook::cancelOrder(uint64_t id)
{
    if (id >= order_lookup.size()) return;
    OrderInfo& info = order_lookup[id];
    if (info.quantity == 0) return;

    if (info.side == 0) bid_levels[info.price].total_qty -= info.quantity;
    else ask_levels[info.price].total_qty -= info.quantity;

    char side_char = (info.side == 0) ? 'B' : 'S';
    UpdateMessage msg{ id, info.price, get_timestamp_raw(), info.quantity, 'X', side_char };
    updateBuffer.push(msg);

    info.quantity = 0; // Lazy cancellation
}

void OrderBook::set_bid_level(uint64_t price)   { bid_bitmap[price / 64] |=  (1ULL << (price % 64)); }
void OrderBook::clear_bid_level(uint64_t price) { bid_bitmap[price / 64] &= ~(1ULL << (price % 64)); }
void OrderBook::set_ask_level(uint64_t price)   { ask_bitmap[price / 64] |=  (1ULL << (price % 64)); }
void OrderBook::clear_ask_level(uint64_t price) { ask_bitmap[price / 64] &= ~(1ULL << (price % 64)); }

uint64_t OrderBook::find_next_bid(uint64_t start_price)
{
    if (start_price == 0) return 0;
    for (int i = static_cast<int>(start_price / 64); i >= 0; --i) {
        uint64_t mask = bid_bitmap[i];
        if (i == static_cast<int>(start_price / 64)) mask &= (1ULL << (start_price % 64 + 1)) - 1;
        if (mask) return static_cast<uint64_t>(i) * 64 + (63 - __builtin_clzll(mask));
    }
    return 0;
}

uint64_t OrderBook::find_next_ask(uint64_t start_price)
{
    for (size_t i = start_price / 64; i < bid_bitmap.size(); ++i) {
        uint64_t mask = ask_bitmap[i];
        if (i == start_price / 64) mask &= ~((1ULL << (start_price % 64)) - 1);
        if (mask) return i * 64 + __builtin_ctzll(mask);
    }
    return MAX_PRICE + 1;
}

void OrderBook::pre_fault_memory()
{
    const size_t page_size = 4096;
    for (size_t i = 0; i < order_lookup.size(); i += page_size / sizeof(OrderInfo)) order_lookup[i].quantity = 0;
    for (size_t i = 0; i < bid_levels.size(); i += page_size / sizeof(PriceLevel)) {
        bid_levels[i].total_qty = 0;
        ask_levels[i].total_qty = 0;
    }
    for (size_t i = 0; i < bid_bitmap.size(); i += page_size / sizeof(uint64_t)) {
        bid_bitmap[i] = 0;
        ask_bitmap[i] = 0;
    }
}