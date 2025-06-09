import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const inputRef = useRef();
  const messagesEndRef = useRef();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function sendMessage() {
    const newMessage = inputRef.current.value.trim();
    if (newMessage) {
      // Add the user's message to the chat
      setMessages(prev => [...prev, { sender: 'user', text: newMessage }]);

      // Clear the input
      setInput('');
      inputRef.current.value = '';

      // Show loading bubble
      setLoading(true);

      // Send the message to the backend
      const response = await fetch('http://localhost:8000/send-message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: newMessage }),
      });

      const data = await response.json();

      // Add the AI's response to the chat
      setMessages(prev => [...prev, { sender: 'ai', text: data.message }]);

      // Hide loading bubble
      setLoading(false);
    }
  }

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((message, index) => (
          <div
            className={`message ${message.sender === 'user' ? 'right' : 'left'}`}
            key={index}
          >
            {message.sender === 'ai' ? (
              <ReactMarkdown>{message.text}</ReactMarkdown>
            ) : (
              message.text
            )}
          </div>
        ))}
        {loading && (
          <div className="message left">
            <span className="loading-bubble"></span>
            <span className="loading-bubble"></span>
            <span className="loading-bubble"></span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <input
          type="text"
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage} aria-label="Send message">
          <Send size={20} />
        </button>
      </div>
    </div>
  );
}

export default App;