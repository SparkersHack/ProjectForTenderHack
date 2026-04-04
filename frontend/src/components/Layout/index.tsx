import React from 'react';
import { useStore } from '../../store/store';
import { ShoppingCart, User as UserIcon, LogOut } from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';

const Layout = ({ children }: { children: React.ReactNode }) => {
    const { user, logout, cart } = useStore();
    const navigate = useNavigate();
    const location = useLocation();
    const cartCount = cart?.totalItems ?? 0;

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <div className="min-h-screen bg-gray-50 font-sans text-gray-900 flex flex-col">
            <header className="bg-white shadow-sm sticky top-0 z-40">
                <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate('/')}>
                        <div className="w-8 h-8 bg-[#E03F3F] rounded flex items-center justify-center text-white font-bold text-xl leading-none">П</div>
                        <div className="font-bold text-xl text-gray-900">Портал Поставщиков</div>
                    </div>

                    <div className="flex items-center gap-6">
                        {location.pathname !== '/login' && (
                            <button
                                onClick={() => navigate('/cart')}
                                className="relative text-gray-600 hover:text-[#E03F3F] transition-colors"
                                title="Корзина"
                            >
                                <ShoppingCart size={22} />
                                {cartCount > 0 && (
                                    <span className="absolute -right-2 -top-2 min-w-[20px] rounded-full bg-[#E03F3F] px-1 text-center text-xs font-bold text-white">
                                        {cartCount}
                                    </span>
                                )}
                            </button>
                        )}
                        {user ? (
                            <div className="flex items-center gap-4">
                                <div className="text-sm text-right hidden md:block">
                                    <div className="font-semibold text-gray-900">ИНН {user.inn}</div>
                                    <div className="text-gray-500">{user.region || 'Регион не определён'}</div>
                                </div>
                                <div className="w-9 h-9 bg-gray-100 rounded-full flex items-center justify-center text-gray-600">
                                    <UserIcon size={18} />
                                </div>
                                <button 
                                    onClick={handleLogout}
                                    className="text-gray-500 hover:text-[#E03F3F] transition-colors ml-2"
                                    title="Выход"
                                >
                                    <LogOut size={20} />
                                </button>
                            </div>
                        ) : (
                            location.pathname !== '/login' && (
                                <button 
                                    onClick={() => navigate('/login')}
                                    className="text-[#E03F3F] font-bold hover:text-red-700 transition-colors"
                                >
                                    Войти
                                </button>
                            )
                        )}
                    </div>
                </div>
            </header>
            
            <main className="flex-grow w-full relative">
                {children}
            </main>
        </div>
    );
};

export default Layout;
