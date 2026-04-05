import React from 'react';
import { useStore } from '../../store/store';
import { Heart, ShoppingBag, Trash2, X } from 'lucide-react';

interface CartProps {
    isOpen: boolean;
    onClose: () => void;
}

const Cart: React.FC<CartProps> = ({ isOpen, onClose }) => {
    const {
        user,
        cartProducts,
        favoriteProducts,
        removeFromCart,
        clearCart,
        removeFromFavorites,
        clearFavorites,
    } = useStore();
    const isSupplier = (user?.entityType?.trim() || '') === 'supplier';
    const products = isSupplier ? favoriteProducts : cartProducts;
    const total = products.reduce((sum, product) => sum + (product.price || 0), 0);
    const title = isSupplier ? 'Избранное' : 'Корзина';
    const emptyTitle = isSupplier ? 'В избранном пока пусто' : 'В корзине пока пусто';
    const primaryActionText = isSupplier ? 'Продолжить поиск' : 'Перейти к покупкам';

    if (!isOpen) {
        return null;
    }

    return (
        <div className="fixed inset-0 z-[100] overflow-hidden">
            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
            <div className="absolute right-0 top-0 h-full w-full max-w-md bg-white shadow-2xl flex flex-col">
                <div className="p-6 border-b border-gray-100 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        {isSupplier ? <Heart className="text-[#da291c]" size={24} /> : <ShoppingBag className="text-[#da291c]" size={24} />}
                        <h2 className="text-xl font-bold">{title}</h2>
                        <span className="bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full text-sm font-medium">
                            {products.length}
                        </span>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
                        <X size={24} />
                    </button>
                </div>

                <div className="flex-grow overflow-y-auto p-6">
                    {products.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-gray-400 gap-4">
                            {isSupplier ? <Heart size={64} strokeWidth={1} /> : <ShoppingBag size={64} strokeWidth={1} />}
                            <p className="text-lg">{emptyTitle}</p>
                            <button
                                onClick={onClose}
                                className="text-[#da291c] font-semibold hover:underline"
                            >
                                {primaryActionText}
                            </button>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {products.map((product) => (
                                <div key={product.id} className="flex gap-4 group">
                                    <div className="flex-grow">
                                        <h3 className="font-bold text-sm text-gray-900 line-clamp-2 mb-1 group-hover:text-[#da291c] transition-colors">
                                            {product.name}
                                        </h3>
                                        <div className="text-xs text-gray-500 mb-2">ID: {product.id}</div>
                                        <div className="flex items-center justify-between">
                                            <div className="font-bold text-[#da291c]">
                                                {product.price.toLocaleString('ru-RU')} ₽
                                            </div>
                                            <button
                                                onClick={() => (isSupplier ? removeFromFavorites(product.id) : removeFromCart(product.id))}
                                                className="text-gray-300 hover:text-red-500 transition-colors"
                                                title="Удалить"
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {products.length > 0 && (
                    <div className="p-6 border-t border-gray-100 bg-gray-50">
                        {isSupplier ? (
                            <div className="flex flex-col gap-3">
                                <div className="text-sm leading-6 text-gray-600">
                                    Здесь сохраняются интересующие товары. Это локальное избранное поставщика, без оформления заказа.
                                </div>
                                <button
                                    onClick={clearFavorites}
                                    className="w-full text-gray-400 text-sm hover:text-gray-600 transition-colors"
                                >
                                    Очистить избранное
                                </button>
                            </div>
                        ) : (
                            <>
                                <div className="flex items-center justify-between mb-6">
                                    <span className="text-gray-600">Итого:</span>
                                    <span className="text-2xl font-bold text-gray-900">
                                        {total.toLocaleString('ru-RU')} ₽
                                    </span>
                                </div>
                                <div className="flex flex-col gap-3">
                                    <button
                                        className="w-full bg-[#da291c] text-white py-4 font-bold hover:bg-red-700 transition-colors shadow-lg shadow-red-200"
                                        onClick={() => {
                                            alert(`Заказ на сумму ${total.toLocaleString('ru-RU')} ₽ успешно оформлен!`);
                                            clearCart();
                                            onClose();
                                        }}
                                    >
                                        Оформить заказ
                                    </button>
                                    <button
                                        onClick={clearCart}
                                        className="w-full text-gray-400 text-sm hover:text-gray-600 transition-colors"
                                    >
                                        Очистить корзину
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Cart;
