#include "order_book.h"
#include "messages.h"
#include "timing.h"
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

void OrderBook::addOrder(uint64_t id, char side, uint64_t price, uint32_t quantity, uint32_t participant_id, OrderType type)
{
    if (price > MAX_PRICE || price == 0 || id >= MAX_ORDERS) // Added ID check
        return;

    uint32_t remaining = quantity;

    if (side == 'B')
    {
        // Match against Asks
        while (remaining > 0 && min_ask_price <= MAX_PRICE && price >= min_ask_price)
        {
            PriceLevel &level = sides[1].levels[min_ask_price];

            // Lazy cancellation cleanup
            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead = level.head;
                level.head = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER)
                    level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_ask_level(min_ask_price);
                min_ask_price = find_next_ask(min_ask_price + 1);
                continue;
            }

            OrderInfo &resting = order_lookup[level.head];

            // Self-Trade Prevention (STP)
            if (participant_id != 0 && resting.participant_id == participant_id)
            {
                uint32_t stp_id = level.head;
                level.total_qty -= resting.quantity;
                resting.quantity = 0;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER)
                    level.tail = NULL_ORDER;
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
            UpdateMessage trade_msg{level.head, min_ask_price, get_timestamp_raw(),
                                    matched,    'T',           'B'};
            updateBuffer.push(trade_msg);

            if (resting.quantity == 0)
            {
                uint32_t filled_id = level.head;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER)
                    level.tail = NULL_ORDER;
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
            OrderInfo &info = order_lookup[id];
            info.set_price_side(static_cast<uint32_t>(price), 0);
            info.quantity = remaining;
            info.participant_id = participant_id;
            info.next_id = NULL_ORDER;

            PriceLevel &level = sides[0].levels[price];
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

            UpdateMessage post_msg{id, price, get_timestamp_raw(), remaining, 'N', 'B'};
            updateBuffer.push(post_msg);
        }
    }
    else // side == 'S'
    {
        while (remaining > 0 && max_bid_price > 0 && price <= max_bid_price)
        {
            PriceLevel &level = sides[0].levels[max_bid_price];

            while (level.head != NULL_ORDER && order_lookup[level.head].quantity == 0)
            {
                uint32_t dead = level.head;
                level.head = order_lookup[dead].next_id;
                if (level.head == NULL_ORDER)
                    level.tail = NULL_ORDER;
                order_lookup[dead].next_id = NULL_ORDER;
            }

            if (level.head == NULL_ORDER || level.total_qty == 0)
            {
                clear_bid_level(max_bid_price);
                max_bid_price = find_next_bid(max_bid_price - 1);
                continue;
            }

            OrderInfo &resting = order_lookup[level.head];

            if (participant_id != 0 && resting.participant_id == participant_id)
            {
                uint32_t stp_id = level.head;
                level.total_qty -= resting.quantity;
                resting.quantity = 0;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER)
                    level.tail = NULL_ORDER;
                order_lookup[stp_id].next_id = NULL_ORDER;
                continue;
            }

            uint32_t matched = std::min(remaining, resting.quantity);
            resting.quantity -= matched;
            remaining -= matched;
            level.total_qty -= matched;
            total_traded_volume += matched;
            traded_notional_value += (static_cast<uint64_t>(matched) * max_bid_price);

            UpdateMessage trade_msg{level.head, max_bid_price, get_timestamp_raw(),
                                    matched,    'T',           'S'};
            updateBuffer.push(trade_msg);

            if (resting.quantity == 0)
            {
                uint32_t filled_id = level.head;
                level.head = resting.next_id;
                if (level.head == NULL_ORDER)
                    level.tail = NULL_ORDER;
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
            OrderInfo &info = order_lookup[id];
            info.set_price_side(static_cast<uint32_t>(price), 1);
            info.quantity = remaining;
            info.participant_id = participant_id;
            info.next_id = NULL_ORDER;

            PriceLevel &level = sides[1].levels[price];
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

            UpdateMessage post_msg{id, price, get_timestamp_raw(), remaining, 'N', 'S'};
            updateBuffer.push(post_msg);
        }
    }
}

void OrderBook::handleCancel(uint64_t id)
{
    if (id >= MAX_ORDERS)
        return;

    // Dispatch via Table (Zero branches, 100% constant time resolving inside template)
    uint8_t side_var = order_lookup[id].get_side();
    (this->*cancel_dispatch[side_var & 1])(id);
}

// Basically find the bitmap entry = price / 64
// and then set the bit corresponding to price % 64
void OrderBook::set_bid_level(uint64_t price)
{
    bid_bitmap[price / 64] |= (1ULL << (price % 64));
}
void OrderBook::clear_bid_level(uint64_t price)
{
    bid_bitmap[price / 64] &= ~(1ULL << (price % 64));
}
void OrderBook::set_ask_level(uint64_t price)
{
    ask_bitmap[price / 64] |= (1ULL << (price % 64));
}
void OrderBook::clear_ask_level(uint64_t price)
{
    ask_bitmap[price / 64] &= ~(1ULL << (price % 64));
}

uint64_t OrderBook::find_next_bid(uint64_t start_price)
{
    if (start_price == 0)
        return 0;

    int i = static_cast<int>(start_price / 64);

    // 1. Handle the FIRST bucket separately (No IF needed!)
    uint64_t mask = bid_bitmap[i];
    mask &= (1ULL << (start_price % 64 + 1)) - 1; // Apply the mask

    if (mask)
    {
        return static_cast<uint64_t>(i) * 64 + (63 - __builtin_clzll(mask));
    }

    // 2. Handle all OTHER buckets in a clean loop (No IF needed!)
    for (--i; i >= 0; --i)
    {
        mask = bid_bitmap[i];
        if (mask)
        {
            return static_cast<uint64_t>(i) * 64 + (63 - __builtin_clzll(mask));
        }
    }
    return 0;
}
uint64_t OrderBook::find_next_ask(uint64_t start_price)
{
    size_t i = start_price / 64;
    if (i >= ask_bitmap.size())
        return MAX_PRICE + 1;

    // 1. Handle the FIRST bucket (The only one that needs masking)
    uint64_t mask = ask_bitmap[i];
    mask &= ~((1ULL << (start_price % 64)) - 1);

    if (mask)
    {
        return i * 64 + __builtin_ctzll(mask);
    }

    // 2. Handle the REST of the buckets (Clean, branchless loop)
    for (++i; i < ask_bitmap.size(); ++i)
    {
        mask = ask_bitmap[i];
        if (mask)
        {
            return i * 64 + __builtin_ctzll(mask);
        }
    }

    return MAX_PRICE + 1;
}

void OrderBook::pre_fault_memory()
{
    const size_t page_size = 4096;
    for (size_t i = 0; i < order_lookup.size(); i += page_size / sizeof(OrderInfo))
        order_lookup[i].quantity = 0;
    for (size_t i = 0; i < sides[0].levels.size(); i += page_size / sizeof(PriceLevel))
    {
        sides[0].levels[i].total_qty = 0;
        sides[1].levels[i].total_qty = 0;
    }
    // for (size_t i = 0; i < bid_bitmap.size(); i += page_size / sizeof(uint64_t))
    // {
    //     bid_bitmap[i] = 0;
    //     ask_bitmap[i] = 0;
    // }
    for (volatile uint64_t &val : bid_bitmap)
        val = 0;
    for (volatile uint64_t &val : ask_bitmap)
        val = 0;
}