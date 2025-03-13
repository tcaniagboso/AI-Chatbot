'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      // TODO: Replace with actual API call
      const response = await new Promise(resolve => 
        setTimeout(() => resolve({ text: "This is a sample response. Replace this with actual API integration." }), 1000)
      );
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: (response as any).text 
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, there was an error processing your request.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-chatbg">
      <div className="max-w-3xl mx-auto py-6 px-4">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`p-6 mb-4 rounded-lg ${
              message.role === 'assistant' ? 'bg-messagebg' : 'bg-chatbg'
            }`}
          >
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 flex items-center justify-center rounded-full bg-gray-700 text-white flex-shrink-0">
                {message.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="flex-1">
                <p className="text-gray-200 whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="fixed bottom-0 left-0 right-0 bg-chatbg border-t border-gray-700 p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="Type your message here..."
            className="w-full p-4 pr-12 rounded-lg bg-inputbg text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={1}
          />
          <button 
            type="submit" 
            className="absolute right-3 bottom-3 p-2 text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isLoading}
          >
            {isLoading ? '⏳' : '➤'}
          </button>
        </form>
      </div>
    </main>
  );
}
