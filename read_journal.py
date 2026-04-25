import struct
import sys

# UpdateMessage struct:
# uint64_t order_id;      (Q)
# uint64_t price;         (Q)
# uint64_t timestamp_raw; (Q)
# uint32_t quantity;      (I)
# char     type;          (c)
# char     side;          (c)
# 2 bytes padding         (xx)
fmt = "QQQIccxx"
struct_size = struct.calcsize(fmt)

def read_journal(filename, limit=20):
    try:
        with open(filename, "rb") as f:
            count = 0
            while True:
                data = f.read(struct_size)
                if not data or len(data) < struct_size:
                    break
                
                order_id, price, ts, qty, msg_type, side = struct.unpack(fmt, data)
                
                # Skip unwritten entries (all zeros)
                if order_id == 0 and price == 0 and qty == 0:
                    continue
                
                msg_type = msg_type.decode('ascii')
                side = side.decode('ascii')
                
                print(f"ID: {order_id:<10} | Price: {price:<6} | Qty: {qty:<6} | Type: {msg_type} | Side: {side} | TS: {ts}")
                
                count += 1
                if limit and count >= limit:
                    break
            
            if count == 0:
                print("No active entries found in the journal.")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")

if __name__ == "__main__":
    limit = 50
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    
    print(f"Reading first {limit} entries from event_journal.bin...\n")
    read_journal("event_journal.bin", limit)
