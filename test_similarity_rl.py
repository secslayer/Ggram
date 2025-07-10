import requests
import json
import time

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000"

# Test authentication
def test_auth():
    print("\n=== Testing Authentication ===")
    auth_data = {
        "username": "testuser",
        "password": "password123"
    }
    response = requests.post(f"{BASE_URL}/api/auth/login", data=auth_data)
    print(f"Authentication: {response.status_code}")
    result = response.json() if response.status_code < 400 else response.text
    print(result)
    return result.get("access_token") if isinstance(result, dict) else None

# Test agent creation for similarity testing
def test_create_agents(token):
    print("\n=== Creating Test Agents for Similarity Testing ===")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create multiple agents with different capabilities and types
    agents = []
    
    # Agent 1: Text generation agent
    agent_data = {
        "name": f"TextAgent_{int(time.time())}",
        "description": "A text generation agent",
        "agent_type": "text",
        "capabilities": ["text-generation", "summarization"],
        "a2a_enabled": True,
        "llm_config": {
            "model": "meta/llama-3.1-8b-instruct",
            "temperature": 0.2
        }
    }
    response = requests.post(f"{BASE_URL}/api/agents/", json=agent_data, headers=headers)
    print(f"Create text agent: {response.status_code}")
    if response.status_code < 400:
        agent = response.json()
        agents.append(agent)
        print(f"Created agent: {agent['name']} with ID: {agent['id']}")
    else:
        print(response.text)
    
    # Wait a bit before creating the next agent
    time.sleep(1)
    
    # Agent 2: QA agent
    agent_data = {
        "name": f"QAAgent_{int(time.time())}",
        "description": "A question answering agent",
        "agent_type": "qa",
        "capabilities": ["question-answering", "information-retrieval"],
        "a2a_enabled": True,
        "llm_config": {
            "model": "meta/llama-3.1-8b-instruct",
            "temperature": 0.1
        }
    }
    response = requests.post(f"{BASE_URL}/api/agents/", json=agent_data, headers=headers)
    print(f"Create QA agent: {response.status_code}")
    if response.status_code < 400:
        agent = response.json()
        agents.append(agent)
        print(f"Created agent: {agent['name']} with ID: {agent['id']}")
    else:
        print(response.text)
    
    # Wait a bit before creating the next agent
    time.sleep(1)
    
    # Agent 3: Code agent
    agent_data = {
        "name": f"CodeAgent_{int(time.time())}",
        "description": "A code generation agent",
        "agent_type": "code",
        "capabilities": ["code-generation", "code-explanation"],
        "a2a_enabled": True,
        "llm_config": {
            "model": "meta/llama-3.1-8b-instruct",
            "temperature": 0.3
        }
    }
    response = requests.post(f"{BASE_URL}/api/agents/", json=agent_data, headers=headers)
    print(f"Create code agent: {response.status_code}")
    if response.status_code < 400:
        agent = response.json()
        agents.append(agent)
        print(f"Created agent: {agent['name']} with ID: {agent['id']}")
    else:
        print(response.text)
    
    # Wait a bit before creating the next agent
    time.sleep(1)
    
    # Agent 4: Similar to Text agent
    agent_data = {
        "name": f"TextAgent2_{int(time.time())}",
        "description": "Another text generation agent",
        "agent_type": "text",
        "capabilities": ["text-generation", "summarization", "paraphrasing"],
        "a2a_enabled": True,
        "llm_config": {
            "model": "meta/llama-3.1-8b-instruct",
            "temperature": 0.2
        }
    }
    response = requests.post(f"{BASE_URL}/api/agents/", json=agent_data, headers=headers)
    print(f"Create text agent 2: {response.status_code}")
    if response.status_code < 400:
        agent = response.json()
        agents.append(agent)
        print(f"Created agent: {agent['name']} with ID: {agent['id']}")
    else:
        print(response.text)
    
    return agents

# Test similarity search with GNN method
def test_similarity_search_gnn(agent_id):
    print("\n=== Testing Similarity Search (GNN) ===")
    # Convert UUID string to integer for API compatibility
    try:
        # Try to convert the UUID to an integer by taking the first part
        numeric_id = int(agent_id.split('-')[0], 16)
    except (ValueError, AttributeError):
        # If conversion fails, use a default value
        numeric_id = 1
        
    search_data = {
        "agent_id": numeric_id,
        "top_k": 3,
        "similarity_threshold": 0.3,
        "method": "gnn"
    }
    response = requests.post(f"{BASE_URL}/api/similarity/search", json=search_data)
    print(f"Similarity search (GNN): {response.status_code}")
    if response.status_code < 400:
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(response.text)

# Test similarity search with Jaccard method
def test_similarity_search_jaccard(agent_id):
    print("\n=== Testing Similarity Search (Jaccard) ===")
    # Convert UUID string to integer for API compatibility
    try:
        # Try to convert the UUID to an integer by taking the first part
        numeric_id = int(agent_id.split('-')[0], 16)
    except (ValueError, AttributeError):
        # If conversion fails, use a default value
        numeric_id = 1
        
    search_data = {
        "agent_id": numeric_id,
        "top_k": 3,
        "similarity_threshold": 0.3,
        "method": "jaccard"
    }
    response = requests.post(f"{BASE_URL}/api/similarity/search", json=search_data)
    print(f"Similarity search (Jaccard): {response.status_code}")
    if response.status_code < 400:
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(response.text)

# Test training the reinforcement learning model
def test_train_rl(token):
    print("\n=== Training Reinforcement Learning Model ===")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/api/similarity/train?num_epochs=50", headers=headers)
    print(f"Train RL model: {response.status_code}")
    if response.status_code < 400:
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(response.text)

# Test collaboration recommendations
def test_collaboration_recommendations(agent_id, token):
    print("\n=== Testing Collaboration Recommendations ===")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Convert UUID string to integer for API compatibility
    try:
        # Try to convert the UUID to an integer by taking the first part
        numeric_id = int(agent_id.split('-')[0], 16)
    except (ValueError, AttributeError):
        # If conversion fails, use a default value
        numeric_id = 1
        
    recommendation_data = {
        "agent_id": numeric_id,
        "top_k": 3
    }
    response = requests.post(f"{BASE_URL}/api/similarity/recommend", json=recommendation_data, headers=headers)
    print(f"Collaboration recommendations: {response.status_code}")
    if response.status_code < 400:
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(response.text)

# Run the tests
if __name__ == "__main__":
    print("Testing Ggram Similarity and Reinforcement Learning...\n")
    
    # Authenticate
    token = test_auth()
    if not token:
        print("Authentication failed, exiting.")
        exit(1)
    
    # Create test agents
    agents = test_create_agents(token)
    if not agents:
        print("Failed to create test agents, exiting.")
        exit(1)
    
    # Use the first agent for testing
    test_agent = agents[0]
    agent_id = test_agent['id']
    
    # Test similarity search
    test_similarity_search_gnn(agent_id)
    test_similarity_search_jaccard(agent_id)
    
    # Train the RL model
    test_train_rl(token)
    
    # Test collaboration recommendations
    test_collaboration_recommendations(agent_id, token)
    
    print("\nAll tests completed!")