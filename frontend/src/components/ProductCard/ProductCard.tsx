import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShoppingCart } from 'lucide-react';

import { addItemToCart, trackEvent } from '../../api';
import { useStore } from '../../store/store';
import { Product } from '../../types';

interface ProductCardProps {
    product: Product;
}

const ProductCard: React.FC<ProductCardProps> = ({ product }) => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const navigate = useNavigate();
    const { user, searchQuery, registerViewedCategory, setCart } = useStore();

    const runtimeUserId = user?.id ?? 'anonymous';

    const handleOpen = async () => {
        registerViewedCategory(product.category);
        try {
            await trackEvent({
                userId: runtimeUserId,
                eventType: 'click',
                steId: product.id,
                category: product.category,
                query: searchQuery,
            });
        } catch (error) {
            console.error('Failed to track click event', error);
        }
        navigate(`/items/${product.id}`);
    };

    const handleAddToCart = async (event: React.MouseEvent) => {
        event.stopPropagation();
        if (!user) {
            navigate('/login');
            return;
        }

        setIsSubmitting(true);
        try {
            const cart = await addItemToCart(user.id, product.id, 1);
            setCart(cart);
        } catch (error) {
            console.error('Failed to add item to cart', error);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div
            onClick={handleOpen}
            className="group relative flex h-full cursor-pointer flex-col rounded-[8px] border border-gray-100 bg-white p-5 transition-shadow hover:shadow-md"
        >
            {product.reasonToShow && (
                <div className="absolute -top-3 right-4 rounded-full border border-blue-100 bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700 shadow-sm">
                    {product.reasonToShow}
                </div>
            )}

            <div className="mb-2 truncate text-xs uppercase tracking-wide text-gray-500">{product.category}</div>

            <h3 className="mb-4 min-h-[50px] text-base font-bold leading-snug text-gray-900 transition-colors group-hover:text-[#E03F3F] md:text-lg">
                {product.name}
            </h3>

            <div className="mb-6 text-xs text-gray-400">ИНН поставщика: {product.supplierInn}</div>

            <div className="mt-auto flex items-end justify-between border-t border-gray-50 pt-4">
                <div className="text-xl font-bold leading-none text-red-600 md:text-2xl">
                    {product.price.toLocaleString('ru-RU')} ₽
                </div>
                <button
                    onClick={handleAddToCart}
                    disabled={isSubmitting}
                    className="flex h-12 w-12 items-center justify-center rounded-[4px] bg-[#E03F3F] text-white shadow-sm transition-colors hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
                    title="Добавить в корзину"
                >
                    <ShoppingCart size={22} />
                </button>
            </div>
        </div>
    );
};

export default ProductCard;
