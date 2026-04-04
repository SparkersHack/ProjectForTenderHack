import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, Loader2, ShoppingCart } from 'lucide-react';
import { addItemToCart, getItem, getOffers, trackEvent } from '../../api';
import { Item, Offer } from '../../types';
import { useStore } from '../../store/store';

const ItemPage = () => {
    const { itemId } = useParams<{ itemId: string }>();
    const navigate = useNavigate();
    const { user, markProductOpen, markProductClose, setCart } = useStore();
    const [item, setItem] = useState<Item | null>(null);
    const [offers, setOffers] = useState<Offer[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isCartSubmitting, setIsCartSubmitting] = useState(false);
    const itemRef = useRef<Item | null>(null);
    const runtimeUserId = user?.id ?? 'anonymous';

    useEffect(() => {
        if (!itemId) {
            navigate('/');
            return;
        }

        let isCancelled = false;
        setIsLoading(true);

        Promise.all([getItem(itemId), getOffers(itemId, user)])
            .then(async ([itemPayload, offersPayload]) => {
                if (isCancelled) {
                    return;
                }
                setItem(itemPayload);
                itemRef.current = itemPayload;
                setOffers(offersPayload.offers);
                markProductOpen(itemPayload.id, itemPayload.category);
                await trackEvent({
                    userId: runtimeUserId,
                    eventType: 'open',
                    steId: itemPayload.id,
                    category: itemPayload.category,
                });
            })
            .catch((error) => {
                console.error('Failed to load item page', error);
            })
            .finally(() => {
                if (!isCancelled) {
                    setIsLoading(false);
                }
            });

        return () => {
            isCancelled = true;
            const currentItem = itemRef.current;
            if (!currentItem) {
                return;
            }
            const isFastBounce = markProductClose(currentItem.id, currentItem.category);
            if (isFastBounce) {
                void trackEvent({
                    userId: runtimeUserId,
                    eventType: 'fast_bounce',
                    steId: currentItem.id,
                    category: currentItem.category,
                }).catch((error) => {
                    console.error('Failed to track fast bounce', error);
                });
            }
        };
    }, [itemId, markProductClose, markProductOpen, navigate, runtimeUserId, user]);

    const handleAddToCart = async () => {
        if (!item) {
            return;
        }
        if (!user) {
            navigate('/login');
            return;
        }
        setIsCartSubmitting(true);
        try {
            const cart = await addItemToCart(user.id, item.id, 1);
            setCart(cart);
            navigate('/cart');
        } catch (error) {
            console.error('Failed to add item to cart', error);
        } finally {
            setIsCartSubmitting(false);
        }
    };

    if (isLoading) {
        return (
            <div className="flex min-h-[60vh] items-center justify-center text-gray-500">
                <Loader2 className="mr-3 animate-spin text-[#E03F3F]" size={28} />
                Загрузка карточки СТЕ...
            </div>
        );
    }

    if (!item) {
        return (
            <div className="mx-auto max-w-[1100px] px-6 py-16">
                <button onClick={() => navigate('/')} className="mb-6 flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900">
                    <ArrowLeft size={16} />
                    Вернуться к поиску
                </button>
                <div className="rounded-[8px] border border-gray-200 bg-white p-8 text-gray-600 shadow-sm">
                    Карточка СТЕ не найдена.
                </div>
            </div>
        );
    }

    return (
        <div className="mx-auto max-w-[1200px] px-6 py-10">
            <button onClick={() => navigate('/')} className="mb-6 flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900">
                <ArrowLeft size={16} />
                Вернуться к поиску
            </button>

            <div className="grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
                <section className="rounded-[8px] border border-gray-200 bg-white p-8 shadow-sm">
                    <div className="mb-3 text-sm uppercase tracking-wide text-gray-500">{item.category}</div>
                    <h1 className="mb-6 text-3xl font-bold leading-tight text-gray-900">{item.name}</h1>

                    <div className="mb-8 grid gap-4 md:grid-cols-2">
                        <div className="rounded-[8px] bg-gray-50 p-4">
                            <div className="text-sm text-gray-500">Минимальная цена</div>
                            <div className="mt-2 text-3xl font-bold text-[#E03F3F]">
                                {item.price.toLocaleString('ru-RU')} ₽
                            </div>
                        </div>
                        <div className="rounded-[8px] bg-gray-50 p-4">
                            <div className="text-sm text-gray-500">Исторических оферт</div>
                            <div className="mt-2 text-3xl font-bold text-gray-900">{item.offerCount}</div>
                        </div>
                    </div>

                    <div className="mb-8 rounded-[8px] border border-gray-200 bg-gray-50 p-5">
                        <div className="mb-3 text-sm font-semibold text-gray-700">Ключевые атрибуты</div>
                        <div className="flex flex-wrap gap-2">
                            {item.attributeKeys.map((attribute) => (
                                <span key={attribute} className="rounded-full bg-white px-3 py-1 text-sm text-gray-700 shadow-sm">
                                    {attribute}
                                </span>
                            ))}
                            {item.attributeKeys.length === 0 && <span className="text-sm text-gray-500">Атрибуты не указаны</span>}
                        </div>
                    </div>

                    <div className="mb-8 rounded-[8px] border border-blue-100 bg-blue-50 p-5 text-sm text-blue-900">
                        Позиция открыта как реальная карточка СТЕ. Поведение пользователя по просмотру и быстрому отказу
                        уходит в runtime events и влияет на последующую выдачу.
                    </div>

                    <button
                        onClick={handleAddToCart}
                        disabled={isCartSubmitting}
                        className="flex items-center gap-3 rounded-[6px] bg-[#E03F3F] px-5 py-3 font-semibold text-white transition-colors hover:bg-red-700 disabled:opacity-60"
                    >
                        <ShoppingCart size={18} />
                        Добавить в корзину
                    </button>
                </section>

                <aside className="rounded-[8px] border border-gray-200 bg-white p-8 shadow-sm">
                    <h2 className="mb-4 text-xl font-bold text-gray-900">Оферты поставщиков</h2>
                    <div className="space-y-4">
                        {offers.map((offer) => (
                            <div key={offer.offerId} className="rounded-[8px] border border-gray-200 p-4">
                                <div className="mb-2 flex items-start justify-between gap-4">
                                    <div>
                                        <div className="font-semibold text-gray-900">ИНН {offer.supplierInn}</div>
                                        <div className="text-sm text-gray-500">
                                            {offer.supplierRegion || 'Регион не указан'}
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-xl font-bold text-[#E03F3F]">
                                            {offer.unitPrice.toLocaleString('ru-RU')} ₽
                                        </div>
                                        <div className="text-xs text-gray-500">score {offer.offerScore.toFixed(2)}</div>
                                    </div>
                                </div>
                                <div className="space-y-2 text-sm text-gray-700">
                                    {offer.explanation.map((line) => (
                                        <div key={line} className="rounded bg-gray-50 px-3 py-2">
                                            {line}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                        {offers.length === 0 && (
                            <div className="rounded-[8px] border border-dashed border-gray-300 p-4 text-sm text-gray-500">
                                Исторические оферты по этой позиции не найдены.
                            </div>
                        )}
                    </div>
                </aside>
            </div>
        </div>
    );
};

export default ItemPage;
