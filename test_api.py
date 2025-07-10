import requests
import json

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000"

# Test the health endpoint
def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health endpoint:", response.status_code)
    print(response.json())
    print()

# Test user creation
def test_create_user():
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "password123"
    }
    response = requests.post(f"{BASE_URL}/api/users/", json=user_data)
    print("Create user:", response.status_code)
    print(response.json() if response.status_code < 400 else response.text)
    print()

# Test authentication
def test_auth():
    auth_data = {
        "username": "testuser",
        "password": "password123"
    }
    response = requests.post(f"{BASE_URL}/api/auth/login", data=auth_data)
    print("Authentication:", response.status_code)
    result = response.json() if response.status_code < 400 else response.text
    print(result)
    print()
    return result.get("access_token") if isinstance(result, dict) else None

# Test agent creation
def test_create_agent(token):
    headers = {"Authorization": f"Bearer {token}"}
    # Use a timestamp to create a unique agent name
    import time
    unique_name = f"TestAgent_{int(time.time())}"
    agent_data = {
        "name": unique_name,
        "description": "A test agent",
        "agent_type": "simple",
        "capabilities": ["text-generation", "question-answering"],
        "a2a_enabled": True,
        "llm_config": {
            "model": "meta/llama-3.1-8b-instruct",
            "temperature": 0.2
        }
    }
    response = requests.post(f"{BASE_URL}/api/agents/", json=agent_data, headers=headers)
    print("Create agent:", response.status_code)
    print(response.json() if response.status_code < 400 else response.text)
    print()

def test_a2a_communication(token):
    headers = {"Authorization": f"Bearer {token}"}

    # Create agent 1
    import time
    agent1_name = f"TestAgent1_{int(time.time())}"
    agent1_data = {
        "name": agent1_name,
        "description": "A test agent 1",
        "agent_type": "simple",
        "capabilities": ["text-generation"],
        "a2a_enabled": True,
        "llm_config": {
            "model": "meta/llama-3.1-8b-instruct",
            "temperature": 0.2
        }
    }
    response = requests.post(f"{BASE_URL}/api/agents/", json=agent1_data, headers=headers)
    print("Create agent 1:", response.status_code)
    agent1 = response.json() if response.status_code < 400 else None
    print(agent1)
    print()

    # Create agent 2
    agent2_name = f"TestAgent2_{int(time.time())}"
    agent2_data = {
        "name": agent2_name,
        "description": "A test agent 2",
        "agent_type": "simple",
        "capabilities": ["text-generation"],
        "a2a_enabled": True,
        "llm_config": {
            "model": "meta/llama-3.1-8b-instruct",
            "temperature": 0.2
        }
    }
    response = requests.post(f"{BASE_URL}/api/agents/", json=agent2_data, headers=headers)
    print("Create agent 2:", response.status_code)
    agent2 = response.json() if response.status_code < 400 else None
    print(agent2)
    print()

    if agent1 and agent2:
        # Send a message from agent 1 to agent 2
        from agents.a2a_client import A2AAgentClient
        import asyncio

        async def run_a2a_test():
            try:
                client = A2AAgentClient(
                    sender_id=agent1["id"],
                    receiver_id=agent2["id"],
                    receiver_endpoint=f"{BASE_URL}/a2a/{agent2['id']}"
                )
                print(f"Sending A2A task from {agent1['id']} to {agent2['id']}")
                result = await client.run_task("Hello from agent 1")
                print("A2A task result:", result)
            except Exception as e:
                print(f"An error occurred during A2A communication: {e}")

        import threading

        def run_in_thread():
            asyncio.run(run_a2a_test())

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

# Run the tests
if __name__ == "__main__":
    print("Testing Ggram API...\n")
    test_health()
    test_create_user()
    token = test_auth()
    if token:
        test_create_agent(token)
        test_a2a_communication(token)
    else:
        print("Authentication failed, skipping agent creation test.")