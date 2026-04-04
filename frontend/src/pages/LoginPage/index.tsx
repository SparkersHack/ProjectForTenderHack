import React, { useState } from 'react';
import { useStore } from '../../store/store';
import { login } from '../../api';
import { useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

const LoginPage = () => {
    const [inn, setInnInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const { setUser } = useStore();
    const navigate = useNavigate();

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!inn.trim()) return;

        setIsLoading(true);
        try {
            const user = await login(inn);
            setUser(user);
            navigate('/');
        } catch (error) {
            console.error("Login failed", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-[calc(100vh-64px)]">
            <div className="bg-white p-10 md:p-12 rounded-[8px] shadow-xl w-full max-w-[440px] border border-transparent">
                <div className="text-center mb-10">
                    <div className="w-16 h-16 bg-[#E03F3F] rounded-[8px] mx-auto flex items-center justify-center text-white font-bold text-4xl mb-6 shadow-md">
                        П
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Вход в систему</h1>
                    <p className="text-gray-500 mt-2 font-medium">Портал Поставщиков B2B</p>
                </div>

                <form onSubmit={handleLogin} className="flex flex-col gap-6">
                    <div>
                        <label className="block text-sm font-semibold text-gray-700 mb-2">ИНН организации или пользователя</label>
                        <input 
                            type="text" 
                            placeholder="Например, 7712345678" 
                            value={inn}
                            onChange={(e) => setInnInput(e.target.value)}
                            required
                            className="w-full px-5 py-4 rounded-[4px] border border-gray-300 focus:border-[#E03F3F] focus:ring-2 focus:ring-[#e03f3f33] outline-none transition-all text-base text-gray-900 bg-gray-50 placeholder-gray-400"
                        />
                    </div>
                    
                    <button 
                        type="submit" 
                        disabled={isLoading || !inn.trim()}
                        className="w-full bg-[#E03F3F] hover:bg-red-700 disabled:opacity-50 text-white font-bold py-4 px-6 rounded-[4px] transition-colors flex items-center justify-center min-h-[56px] text-lg shadow-md"
                    >
                        {isLoading ? <Loader2 size={24} className="animate-spin" /> : "Войти"}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default LoginPage;
