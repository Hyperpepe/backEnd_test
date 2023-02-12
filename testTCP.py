import socket
import json

# Server configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 50000

# Create socket and connect to SERVER_IP and SERVER_PORT
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

# Create JSON data
data = {"message": "Hello from the client!"}

# Encode JSON data as bytes
json_data = json.dumps(data).encode("utf-8")

# Send data to the server
client_socket.sendall(json_data)
print("Sent data: {}".format(data))

# Receive response from the server
response = client_socket.recv(1024).decode("utf-8")
print("Received response: {}".format(response))

# Close the connection
client_socket.close()
