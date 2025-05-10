// Chatbot functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chat history
    let chatHistory = JSON.parse(localStorage.getItem('njChatHistory')) || [];
    let currentChatId = null;
    let currentChat = [];
    
    // DOM Elements
    const chatBox = document.getElementById('chat-box');
    const historyList = document.getElementById('history-list');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistoryElement = document.getElementById('chat-history');
    const hideHistoryBtn = document.getElementById('hide-history');
    const showHistoryBtn = document.getElementById('show-history-btn');
    
    // Initialize
    loadPresetQuestions();
    setupEventListeners();
    renderChatHistory();
    
    function setupEventListeners() {
        // Enter key submission
        promptInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                generateResponse();
            }
        });
        
        // Send button click
        sendBtn.addEventListener('click', generateResponse);
        
        // Preset question buttons
        document.querySelectorAll('.preset-question-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                promptInput.value = this.dataset.question;
                promptInput.focus();
            });
        });
        
        // Chat history toggle
        hideHistoryBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            chatHistoryElement.classList.add('collapsed');
            showHistoryBtn.style.display = 'block';
        });
        
        showHistoryBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            chatHistoryElement.classList.remove('collapsed');
            showHistoryBtn.style.display = 'none';
        });
    }
    
    function renderChatHistory() {
        if (chatHistory.length === 0) {
            historyList.innerHTML = '<div class="text-muted text-center py-3">No chat history yet</div>';
            return;
        }
        
        historyList.innerHTML = '';
        chatHistory.slice(0, 5).forEach((chat, index) => {
            const firstMessage = chat.messages[0]?.content || 'New chat';
            
            const historyItem = document.createElement('div');
            historyItem.className = `history-item ${chat.id === currentChatId ? 'active' : ''}`;
            historyItem.innerHTML = `
                <div class="history-content">
                    <div class="history-title">Chat ${chatHistory.length - index}</div>
                    <div class="history-preview">${firstMessage.substring(0, 30)}${firstMessage.length > 30 ? '...' : ''}</div>
                </div>
                <button class="delete-history" data-id="${chat.id}" title="Delete this chat">
                    <i class="fas fa-trash"></i>
                </button>
            `;
            
            // Click to load chat
            historyItem.querySelector('.history-content').addEventListener('click', () => {
                loadChat(chat.id);
                chatHistoryElement.classList.remove('collapsed');
                showHistoryBtn.style.display = 'none';
            });
            
            // Delete button click
            historyItem.querySelector('.delete-history').addEventListener('click', function(e) {
                e.stopPropagation();
                const chatId = this.dataset.id;
                showDeleteConfirmation(chatId);
            });
            
            historyList.appendChild(historyItem);
        });
    }
    
    function showDeleteConfirmation(chatId) {
        const modal = new bootstrap.Modal(document.getElementById('confirmDeleteModal'));
        modal.show();
        
        document.getElementById('confirmDeleteBtn').onclick = function() {
            deleteChatHistory(chatId);
            modal.hide();
        };
    }
    
    function deleteChatHistory(chatId) {
        chatHistory = chatHistory.filter(c => c.id !== chatId);
        localStorage.setItem('njChatHistory', JSON.stringify(chatHistory));
        
        if (currentChatId === chatId) {
            startNewChat();
        }
        
        renderChatHistory();
    }
    
    function loadChat(chatId) {
        const chat = chatHistory.find(c => c.id === chatId);
        if (!chat) return;
        
        currentChatId = chatId;
        currentChat = chat.messages;
        renderChat();
        renderChatHistory();
    }
    
    function startNewChat() {
        currentChatId = Date.now().toString();
        currentChat = [];
        renderChat();
    }
    
    function renderChat() {
        chatBox.innerHTML = '';
        
        if (currentChat.length === 0) {
            chatBox.innerHTML = `
                <div class="alert alert-info">
                    <strong>Welcome!</strong> I'm your New Jersey travel assistant. Ask me about attractions, food, or travel tips!
                </div>
                <p class="text-muted">Try asking me:</p>
                <div class="preset-question mb-2 p-2 bg-light rounded">
                    <span>What are the top attractions in Atlantic City?</span>
                    <button class="btn btn-sm btn-outline-primary float-end preset-question-btn" data-question="What are the top attractions in Atlantic City?">Use</button>
                </div>
            `;
            
            document.querySelectorAll('.preset-question-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    promptInput.value = this.dataset.question;
                    promptInput.focus();
                });
            });
            
            return;
        }
        
        currentChat.forEach(msg => {
            if (msg.role === 'user') {
                chatBox.innerHTML += `
                    <div class="user-message">
                        <div class="message-bubble">
                            <strong>You:</strong> ${msg.content}
                        </div>
                    </div>
                `;
            } else {
                chatBox.innerHTML += `
                    <div class="ai-message">
                        <div class="message-bubble">
                            <strong>AI:</strong> ${msg.content}
                        </div>
                    </div>
                `;
            }
        });
        
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    function saveChat() {
        if (currentChat.length === 0) return;
        
        // Remove if already exists
        chatHistory = chatHistory.filter(c => c.id !== currentChatId);
        
        // Add to beginning of array
        chatHistory.unshift({
            id: currentChatId,
            timestamp: new Date().toISOString(),
            messages: currentChat
        });
        
        // Keep only last 5 chats
        chatHistory = chatHistory.slice(0, 5);
        
        localStorage.setItem('njChatHistory', JSON.stringify(chatHistory));
        renderChatHistory();
    }
    
    function loadPresetQuestions() {
        if (currentChat.length === 0) {
            startNewChat();
        }
    }
    
    async function generateResponse() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;
        
        // Start new chat if none exists
        if (!currentChatId) {
            startNewChat();
        }
        
        promptInput.disabled = true;
        sendBtn.disabled = true;
        
        // Add user message
        currentChat.push({
            role: 'user',
            content: prompt,
            timestamp: new Date().toISOString()
        });
        
        // Clear chat box before showing new messages
        chatBox.innerHTML = '';
        renderChat();
        
        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<strong>AI:</strong> Thinking...';
        chatBox.appendChild(typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        try {
            // Send request to backend
            const response = await fetch('http://localhost:8000/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: prompt })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add AI response
            currentChat.push({
                role: 'assistant',
                content: data.response,
                timestamp: new Date().toISOString()
            });
            
            // Save to history
            saveChat();
        } catch (error) {
            console.error('Error:', error);
            currentChat.push({
                role: 'assistant',
                content: "Sorry, something went wrong. Please try again.",
                timestamp: new Date().toISOString()
            });
        } finally {
            // Re-render chat to show both messages
            renderChat();
            
            // Re-enable input
            promptInput.disabled = false;
            sendBtn.disabled = false;
            promptInput.value = '';
            promptInput.focus();
        }
    }
});

// Back to top button functionality
document.getElementById('backToTop')?.addEventListener('click', function(e) {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
});