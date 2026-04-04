import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, ReceiptText } from 'lucide-react';
import { createProcurement, getCart } from '../../api';
import { useStore } from '../../store/store';

const CartPage = () => {
    const navigate = useNavigate();
    const { user, cart, setCart, lastProcurement, setLastProcurement } = useStore();
    const [isLoading, setIsLoading] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        if (!user) {
            return;
        }
        setIsLoading(true);
        getCart(user.id)
            .then((payload) => {
                setCart(payload);
            })
            .catch((error) => {
                console.error('Failed to load cart', error);
            })
            .finally(() => {
                setIsLoading(false);
            });
    }, [setCart, user]);

    const handleCreateProcurement = async () => {
        if (!user) {
            navigate('/login');
            return;
        }
        setIsSubmitting(true);
        try {
            const procurement = await createProcurement(user.id, 'direct_purchase');
            setLastProcurement(procurement);
            setCart({
                userId: user.id,
                items: [],
                totalItems: 0,
                totalAmount: 0,
            });
        } catch (error) {
            console.error('Failed to create procurement', error);
        } finally {
            setIsSubmitting(false);
        }
    };

    if (!user) {
        return (
            <div className="mx-auto max-w-[900px] px-6 py-16">
                <div className="rounded-[8px] border border-gray-200 bg-white p-8 shadow-sm">
                    <h1 className="mb-3 text-2xl font-bold text-gray-900">Корзина доступна после входа</h1>
                    <p className="mb-6 text-gray-600">Для формирования закупки нужен идентификатор заказчика.</p>
                    <button
                        onClick={() => navigate('/login')}
                        className="rounded-[6px] bg-[#E03F3F] px-5 py-3 font-semibold text-white transition-colors hover:bg-red-700"
                    >
                        Перейти ко входу
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="mx-auto max-w-[1100px] px-6 py-10">
            <div className="mb-8 flex items-end justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Корзина закупки</h1>
                    <div className="mt-2 text-sm text-gray-500">Заказчик: ИНН {user.inn}</div>
                </div>
                <button
                    onClick={() => navigate('/')}
                    className="rounded-[6px] border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:border-gray-300 hover:text-gray-900"
                >
                    Вернуться к поиску
                </button>
            </div>

            {lastProcurement && (
                <div className="mb-6 rounded-[8px] border border-green-200 bg-green-50 p-5 text-green-900">
                    <div className="font-semibold">Закупка создана</div>
                    <div className="mt-1 text-sm">
                        ID: {lastProcurement.procurementId}, сумма {lastProcurement.totalAmount.toLocaleString('ru-RU')} ₽
                    </div>
                </div>
            )}

            {isLoading ? (
                <div className="flex min-h-[40vh] items-center justify-center text-gray-500">
                    <Loader2 className="mr-3 animate-spin text-[#E03F3F]" size={28} />
                    Загрузка корзины...
                </div>
            ) : (
                <div className="grid gap-8 lg:grid-cols-[1.4fr_0.8fr]">
                    <section className="rounded-[8px] border border-gray-200 bg-white p-8 shadow-sm">
                        <h2 className="mb-4 text-xl font-bold text-gray-900">Позиции</h2>
                        <div className="space-y-4">
                            {cart?.items.map((item) => (
                                <div key={item.steId} className="rounded-[8px] border border-gray-200 p-4">
                                    <div className="mb-2 flex items-start justify-between gap-4">
                                        <div>
                                            <div className="font-semibold text-gray-900">{item.name}</div>
                                            <div className="text-sm text-gray-500">{item.category}</div>
                                        </div>
                                        <div className="text-right">
                                            <div className="font-semibold text-[#E03F3F]">
                                                {(item.price * item.quantity).toLocaleString('ru-RU')} ₽
                                            </div>
                                            <div className="text-sm text-gray-500">Кол-во: {item.quantity}</div>
                                        </div>
                                    </div>
                                    <div className="text-sm text-gray-500">ИНН поставщика: {item.supplierInn}</div>
                                </div>
                            ))}
                            {!cart || cart.items.length === 0 ? (
                                <div className="rounded-[8px] border border-dashed border-gray-300 p-5 text-sm text-gray-500">
                                    Корзина пуста. Добавьте позиции из карточки СТЕ или из результатов поиска.
                                </div>
                            ) : null}
                        </div>
                    </section>

                    <aside className="rounded-[8px] border border-gray-200 bg-white p-8 shadow-sm">
                        <div className="mb-6 flex items-center gap-3">
                            <ReceiptText className="text-[#E03F3F]" size={24} />
                            <h2 className="text-xl font-bold text-gray-900">Сводка закупки</h2>
                        </div>

                        <div className="space-y-3 text-sm text-gray-600">
                            <div className="flex items-center justify-between">
                                <span>Позиций</span>
                                <span className="font-semibold text-gray-900">{cart?.totalItems ?? 0}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span>Сумма</span>
                                <span className="font-semibold text-gray-900">
                                    {(cart?.totalAmount ?? 0).toLocaleString('ru-RU')} ₽
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span>Тип закупки</span>
                                <span className="font-semibold text-gray-900">direct_purchase</span>
                            </div>
                        </div>

                        <button
                            onClick={handleCreateProcurement}
                            disabled={isSubmitting || !cart || cart.totalItems === 0}
                            className="mt-8 w-full rounded-[6px] bg-[#E03F3F] px-5 py-3 font-semibold text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            {isSubmitting ? 'Создание...' : 'Создать закупку'}
                        </button>
                    </aside>
                </div>
            )}
        </div>
    );
};

export default CartPage;
