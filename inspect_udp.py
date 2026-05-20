import socket
import struct

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print("Listening...")
data, addr = sock.recvfrom(4096)
print(f"Received {len(data)} bytes")
print("Hex dump of first 32 bytes:")
print(data[:32].hex())

# Let's try to find 256 (0x0100 -> 00 01) and 44100 (0xAC44 -> 44 AC)
# And the node_id which is likely 'A' (0x41) or 'B' (0x42)
