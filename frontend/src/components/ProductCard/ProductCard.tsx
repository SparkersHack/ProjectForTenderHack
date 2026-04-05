import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Product } from '../../types';
import { useStore } from '../../store/store';
import { Check, ChevronDown, ShoppingCart, X } from 'lucide-react';

interface ProductCardProps {
    product: Product;
}

const ProductCard: React.FC<ProductCardProps> = ({ product }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isCartFeedbackVisible, setIsCartFeedbackVisible] = useState(false);
    const isOpenRef = useRef(false);
    const wasInCartRef = useRef(false);
    const { simulateProductOpen, simulateProductClose, addToCart, cartProducts } = useStore();
    const isInCart = cartProducts.some((item) => item.id === product.id);

    useEffect(() => {
        if (isInCart && !wasInCartRef.current) {
            setIsCartFeedbackVisible(true);
        }
        if (!isInCart) {
            setIsCartFeedbackVisible(false);
        }
        wasInCartRef.current = isInCart;
    }, [isInCart]);

    useEffect(() => {
        if (!isCartFeedbackVisible) {
            return;
        }
        const timeoutId = window.setTimeout(() => setIsCartFeedbackVisible(false), 1600);
        return () => window.clearTimeout(timeoutId);
    }, [isCartFeedbackVisible]);

    const handleOpen = () => {
        setIsOpen(true);
        simulateProductOpen(product.id, product.category);
    };

    const handleAddToCart = (e: React.MouseEvent) => {
        e.stopPropagation();
        addToCart(product);
    };

    const handleClose = (e: React.MouseEvent) => {
        e.stopPropagation();
        setIsOpen(false);
        simulateProductClose(product.id, product.category);
    };

    useEffect(() => {
        isOpenRef.current = isOpen;
    }, [isOpen]);

    useEffect(() => {
        return () => {
            if (isOpenRef.current) {
                simulateProductClose(product.id, product.category);
            }
        };
    }, [product.id, product.category, simulateProductClose]);

    useEffect(() => {
        if (!isOpen) {
            return;
        }
        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = previousOverflow;
        };
    }, [isOpen]);

    return (
        <>
            <div 
                onClick={handleOpen}
                className={`bg-white border shadow-[0_8px_24px_rgba(15,23,42,0.05)] hover:shadow-[0_14px_32px_rgba(15,23,42,0.08)] transition-all cursor-pointer relative h-full flex flex-col group ${
                    isInCart ? 'border-[#c7efd2]' : 'border-[#e6e8ec]'
                } ${
                    isCartFeedbackVisible ? 'ring-2 ring-[#86efac] shadow-[0_18px_36px_rgba(34,197,94,0.16)]' : ''
                }`}
            >
                <div className="p-5 flex flex-col flex-1">
                    {product.reasonToShow && (
                        <div className="mb-3 inline-flex w-fit bg-[#eef5ff] text-[#356aa6] border border-[#dce9fb] text-xs font-semibold px-3 py-1">
                            {product.reasonToShow}
                        </div>
                    )}

                    <h3 className="text-[#2f3640] font-semibold text-[17px] mb-5 line-clamp-3 min-h-[82px] leading-[1.35] group-hover:text-[#d63d2b] transition-colors">
                        {product.name}
                    </h3>

                    <div className="space-y-1.5 text-[15px] leading-6 text-[#2f3640]">
                        <div>
                            <span className="font-semibold">ID СТЕ </span>
                            <span className="text-[#3e6ea7]">{product.id}</span>
                        </div>
                        <div>
                            <span className="font-semibold">Предложений: </span>
                            <span className="text-[#3e6ea7]">{product.offerCount}</span>
                        </div>
                        <div>
                            <span className="font-semibold">Поставщик: </span>
                            <span className="text-[#3e6ea7] break-all">{product.supplierInn || 'не указан'}</span>
                        </div>
                        <div>
                            <span className="font-semibold">Категория: </span>
                            <span className="text-[#3e6ea7]">{product.category}</span>
                        </div>
                    </div>

                    <button
                        type="button"
                        onClick={(e) => {
                            e.stopPropagation();
                            handleOpen();
                        }}
                        className="mt-4 inline-flex items-center gap-1.5 text-[15px] font-medium text-[#2f3640] underline underline-offset-4 decoration-dotted hover:text-[#d63d2b] transition-colors"
                    >
                        Характеристики
                        <ChevronDown size={15} />
                    </button>

                    <div className="mt-auto pt-6">
                        <div className="border-t border-[#e8eaef] pt-4 flex items-center justify-between gap-3">
                            <div className="text-[#1f2937] font-bold text-[18px] md:text-[20px] leading-none">
                                {product.price > 0
                                    ? `${product.price.toLocaleString('ru-RU')} ₽`
                                    : <span className="text-[#9ca3af] text-[15px] font-medium">Цена не указана</span>
                                }
                            </div>
                            <button
                                type="button"
                                onClick={handleAddToCart}
                                className={`flex-shrink-0 rounded-[2px] border p-2 transition-all ${
                                    isInCart
                                        ? 'border-[#8dd7a5] bg-[#ecfdf3] text-[#16723e]'
                                        : 'border-[#d63d2b] text-[#d63d2b] hover:bg-[#fdf2f1]'
                                } ${isCartFeedbackVisible ? 'animate-pulse' : ''}`}
                                title={isInCart ? 'Товар уже в корзине' : 'Добавить в корзину'}
                            >
                                {isInCart ? <Check size={18} /> : <ShoppingCart size={18} />}
                            </button>
                        </div>
                        {isInCart && (
                            <div
                                className={`mt-3 inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${
                                    isCartFeedbackVisible
                                        ? 'border-[#86efac] bg-[#ecfdf3] text-[#16723e] animate-pulse'
                                        : 'border-[#d6eadb] bg-[#f4fbf6] text-[#2e6d46]'
                                }`}
                            >
                                <Check size={12} />
                                {isCartFeedbackVisible ? 'Добавлено в корзину' : 'Уже в корзине'}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {isOpen && typeof document !== 'undefined' && createPortal(
                <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-[9999] p-4 backdrop-blur-sm" onClick={handleClose}>
                    <div 
                        className="bg-white w-full max-w-xl rounded-[4px] relative flex flex-col shadow-2xl overflow-hidden"
                        onClick={e => e.stopPropagation()}
                    >
                        <button onClick={handleClose} className="absolute top-4 right-4 text-gray-400 hover:text-gray-900 transition-colors">
                            <X size={28} />
                        </button>

                        <div className="p-8 pt-12">
                            <div className="text-sm text-gray-500 mb-2 uppercase tracking-wide">{product.category}</div>
                            <h2 className="text-2xl md:text-[30px] font-bold mb-6 text-gray-900 leading-tight">{product.name}</h2>

                            <div className="space-y-2 text-[15px] leading-6 text-[#2f3640] mb-6">
                                <div>
                                    <span className="font-semibold">ID СТЕ </span>
                                    <span className="text-[#3e6ea7]">{product.id}</span>
                                </div>
                                <div>
                                    <span className="font-semibold">Предложений: </span>
                                    <span className="text-[#3e6ea7]">{product.offerCount}</span>
                                </div>
                                <div>
                                    <span className="font-semibold">Поставщик: </span>
                                    <span className="text-[#3e6ea7]">{product.supplierInn || 'не указан'}</span>
                                </div>
                            </div>

                            {product.descriptionPreview && (
                                <div className="bg-gray-50 border border-gray-200 p-4 mb-6 text-gray-700 text-sm leading-relaxed">
                                    {product.descriptionPreview}
                                </div>
                            )}
                            
                            <div className="bg-blue-50 border border-blue-100 p-4 mb-8 text-blue-800 text-sm">
                                <p className="font-semibold mb-1">Режим имитации пользовательского опыта</p>
                                Закройте это окно <strong>быстрее, чем за 2 секунды</strong>, чтобы система засчитала быстрый отказ. В результате эта категория товаров будет реже появляться в ленте.
                            </div>

                            <div className="flex items-end justify-between gap-4">
                                <div className="text-[#1f2937] font-bold text-3xl md:text-4xl">
                                    {product.price.toLocaleString('ru-RU')} ₽
                                </div>
                                <button
                                    type="button"
                                    onClick={handleAddToCart}
                                    className={`inline-flex items-center gap-2 px-5 py-3 font-semibold transition-all ${
                                        isInCart
                                            ? 'bg-[#15803d] text-white shadow-[0_12px_24px_rgba(34,197,94,0.24)]'
                                            : 'bg-[#da291c] hover:bg-[#bf2418] text-white'
                                    } ${isCartFeedbackVisible ? 'animate-pulse' : ''}`}
                                >
                                    {isInCart ? <Check size={18} /> : <ShoppingCart size={18} />}
                                    {isInCart ? 'Уже в корзине' : 'Добавить в корзину'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>,
                document.body
            )}
        </>
    );
};

export default ProductCard;
