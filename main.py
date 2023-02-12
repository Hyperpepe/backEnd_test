import socket
import json

# Server configuration
IP = "192.168.137.1"
PORT = 50000

# Create socket and bind to IP and PORT
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((IP, PORT))

# Listen for incoming connections
server_socket.listen(1)
print("Listening on {}:{}".format(IP, PORT))

while True:
    # Accept incoming connection
    client_socket, client_address = server_socket.accept()
    print("Accepted connection from {}:{}".format(*client_address))

    # Receive data from the client
    data = client_socket.recv(1024).decode("utf-8")
    print("Received data: {}".format(data))

    # Decode JSON data
    json_data = json.loads(data)
    print("JSON data:", json_data)

    # Send response to the client
    client_socket.sendall("ACK".encode("utf-8"))

    # Close the connection
    client_socket.close()