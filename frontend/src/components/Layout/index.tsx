import React, { useEffect, useRef, useState } from 'react';
import { useStore } from '../../store/store';
import { User as UserIcon, LogOut, ChevronDown, Heart, ShoppingCart } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Cart from '../Cart';
import portalSuppliersLogo from '../../assets/portal-suppliers-logo.jpeg';

const ORGANIZATION_TYPE_DETAILS: Record<string, { title: string; scope: string }> = {
    supplier: {
        title: 'У этого ИНН открыт профиль поставщика.',
        scope: 'Поиск и персонализация строятся по истории поставок, категориям и товарам этого поставщика.',
    },
    healthcare: {
        title: 'Подходит для медицинских учреждений и служб здравоохранения.',
        scope: 'Больницы, поликлиники, диспансеры, амбулатории, аптеки и другие медорганизации.',
    },
    education: {
        title: 'Подходит для образовательных учреждений.',
        scope: 'Школы, гимназии, лицеи, детские сады, колледжи, техникумы и вузы.',
    },
    office_admin: {
        title: 'Подходит для административного и офисного профиля.',
        scope: 'Администрации, департаменты, комитеты, инспекции, архивы и типовые офисные подразделения.',
    },
    facilities: {
        title: 'Подходит для хозяйственных и эксплуатационных организаций.',
        scope: 'Коммунальные службы, благоустройство, дорожные, ремонтные и эксплуатационные подразделения.',
    },
    security_it: {
        title: 'Подходит для ИТ-, связи и безопасности.',
        scope: 'Службы информатизации, связи, охраны, пожарной и информационной безопасности.',
    },
    general: {
        title: 'У этой организации смешанный профиль без одного доминирующего типа.',
        scope: 'Тип не сводится к одной устойчивой группе, поэтому персонализация строится по общему профилю.',
    },
};

const Layout = ({ children }: { children: React.ReactNode }) => {
    const { user, logout, cartProducts, favoriteProducts } = useStore();
    const navigate = useNavigate();
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const [isCartOpen, setIsCartOpen] = useState(false);
    const [visibleFrequentProductsCount, setVisibleFrequentProductsCount] = useState(6);
    const profileRef = useRef<HTMLDivElement | null>(null);

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    useEffect(() => {
        if (!isProfileOpen) return;

        const handleClickOutside = (event: MouseEvent) => {
            if (!profileRef.current?.contains(event.target as Node)) {
                setIsProfileOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isProfileOpen]);

    useEffect(() => {
        setVisibleFrequentProductsCount(6);
    }, [isProfileOpen, user?.inn]);

    const frequentProducts = user?.frequentProducts ?? [];
    const hasHiddenFrequentProducts = frequentProducts.length > visibleFrequentProductsCount;
    const canCollapseFrequentProducts =
        frequentProducts.length > 6 && visibleFrequentProductsCount >= frequentProducts.length;
    const entityType = user?.entityType?.trim() || 'customer';
    const isSupplier = entityType === 'supplier';
    const organizationTypeCode = user?.organizationTypeCode?.trim() || 'general';
    const organizationTypeLabel = user?.organizationTypeLabel?.trim() || 'Общий профиль';
    const organizationTypeSource = user?.organizationTypeSource?.trim() || 'По истории закупок';
    const entityName = user?.customerName?.trim() || `ИНН ${user?.inn ?? ''}`;
    const organizationTypeDetails = ORGANIZATION_TYPE_DETAILS[organizationTypeCode] ?? ORGANIZATION_TYPE_DETAILS.general;
    const profileTitle = isSupplier ? 'Профиль поставщика' : 'Профиль заказчика';
    const entityBadgeLabel = isSupplier ? 'Поставщик' : 'Заказчик';
    const historyLabel = isSupplier
        ? `Персонализация поиска строится по истории поставщика с ИНН ${user?.inn}`
        : `Персонализация поиска строится по истории ИНН ${user?.inn}`;
    const profileKindTitle = isSupplier ? 'Тип профиля' : 'Тип организации';
    const topCategoriesTitle = isSupplier ? 'Топ-категории поставщика' : 'Топ-категории';
    const frequentProductsTitle = isSupplier ? 'Часто поставлялось' : 'Часто закупалось';
    const savedItemsCount = isSupplier ? favoriteProducts.length : cartProducts.length;
    const savedItemsTitle = isSupplier ? 'Избранное' : 'Корзина';

    return (
        <div className="min-h-screen bg-[#f6f7f9] font-sans text-gray-900 flex flex-col">
            <header className="bg-white border-b border-[#eceef1] sticky top-0 z-40">
                <div className="max-w-[1600px] mx-auto px-8 lg:px-12 h-[108px] flex items-center justify-between">
                    {/* Logo */}
                    <button
                        type="button"
                        className="flex items-center gap-5"
                        onClick={() => navigate('/')}
                    >
                        <img
                            src={portalSuppliersLogo}
                            alt="Портал поставщиков"
                            className="h-[72px] md:h-[80px] w-auto object-contain"
                        />
                    </button>

                    {/* Right side controls */}
                    <div className="flex items-center gap-4">
                        {user ? (
                            <div className="flex items-center gap-3">
                                {/* Profile dropdown */}
                                <div className="relative" ref={profileRef}>
                                    <button
                                        onClick={() => setIsProfileOpen((v) => !v)}
                                        className="flex items-center gap-3 rounded-[2px] border border-[#e6e8ec] bg-white px-4 py-2.5 hover:border-[#d9dde3] transition-colors"
                                    >
                                        <div className="text-sm text-right hidden md:block">
                                            <div className="flex items-center justify-end gap-2 font-semibold text-gray-900">
                                                <span>{profileTitle} / ИНН {user.inn}</span>
                                                <span className="inline-flex items-center rounded-full border border-[#ffe0db] bg-[#fff3ef] px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#ba4b2a]">
                                                    {entityBadgeLabel}
                                                </span>
                                            </div>
                                            <div className="text-xs text-gray-500">
                                                {user.region || 'Регион не определён'}
                                            </div>
                                        </div>
                                        <div className="w-9 h-9 bg-gray-100 rounded-full flex items-center justify-center text-gray-600">
                                            <UserIcon size={18} />
                                        </div>
                                        <ChevronDown
                                            size={16}
                                            className={`text-gray-400 transition-transform ${isProfileOpen ? 'rotate-180' : ''}`}
                                        />
                                    </button>

                                    {isProfileOpen && (
                                        <div className="absolute right-0 mt-3 w-[420px] max-w-[calc(100vw-32px)] max-h-[min(78vh,680px)] bg-white border border-gray-200 rounded-[8px] shadow-xl p-5 z-50 overflow-y-auto">
                                            <div className="mb-5">
                                                <div className="flex items-center gap-2">
                                                    <div className="text-sm font-semibold text-gray-900">
                                                        {profileTitle}
                                                    </div>
                                                    <span className="inline-flex items-center rounded-full border border-[#ffe0db] bg-[#fff3ef] px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#ba4b2a]">
                                                        {entityBadgeLabel}
                                                    </span>
                                                </div>
                                                <div className="text-xs text-gray-500 mt-1">
                                                    {historyLabel}
                                                </div>
                                            </div>

                                            <div className="mb-5 rounded-[10px] border border-[#e6ebf0] bg-[#f8fafc] p-4">
                                                <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">
                                                    {profileKindTitle}
                                                </div>
                                                <div className="text-sm font-semibold text-gray-900 leading-snug">
                                                    {entityName}
                                                </div>
                                                <div className="mt-3 flex flex-wrap gap-2">
                                                    <span className="inline-flex items-center rounded-full border border-[#ffe0db] bg-[#fff3ef] px-3 py-1 text-sm font-semibold text-[#ba4b2a]">
                                                        {entityBadgeLabel}
                                                    </span>
                                                    {!isSupplier && (
                                                        <span className="inline-flex items-center rounded-full border border-[#d7e3f8] bg-[#eef5ff] px-3 py-1 text-sm font-semibold text-[#2f5e9b]">
                                                            {organizationTypeLabel}
                                                        </span>
                                                    )}
                                                    <span className="inline-flex items-center rounded-full border border-[#ebeef2] bg-white px-3 py-1 text-sm text-[#556070]">
                                                        {user.region || 'Регион не определён'}
                                                    </span>
                                                </div>
                                                <div className="mt-2 text-xs text-gray-500">
                                                    {organizationTypeSource}
                                                </div>
                                                <div className="mt-4 rounded-[10px] border border-[#e3e8ef] bg-white px-4 py-3">
                                                    <div className="text-[11px] uppercase tracking-[0.12em] text-gray-400">
                                                        Что означает этот тип
                                                    </div>
                                                    <div className="mt-2 text-sm font-semibold text-[#1f2937]">
                                                        {organizationTypeDetails.title}
                                                    </div>
                                                    <div className="mt-1 text-sm leading-6 text-[#5b6574]">
                                                        {organizationTypeDetails.scope}
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="mb-5">
                                                <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">
                                                    {topCategoriesTitle}
                                                </div>
                                                <div className="flex flex-wrap gap-2">
                                                    {(user.topCategories ?? []).slice(0, 5).map((item) => (
                                                        <div
                                                            key={item.category}
                                                            className="px-3 py-2 rounded-full bg-[#fff5f5] border border-[#ffd6d6] text-sm text-gray-800"
                                                        >
                                                            <span className="font-semibold">{item.category}</span>
                                                            <span className="ml-2 text-gray-500">{item.purchaseCount}</span>
                                                        </div>
                                                    ))}
                                                    {(user.topCategories ?? []).length === 0 && (
                                                        <div className="text-sm text-gray-500">Категории не найдены.</div>
                                                    )}
                                                </div>
                                            </div>

                                            <div>
                                                <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">
                                                    {frequentProductsTitle}
                                                </div>
                                                <div className="space-y-3 pr-1">
                                                    {frequentProducts.slice(0, visibleFrequentProductsCount).map((item) => (
                                                        <div
                                                            key={item.steId}
                                                            className="pb-3 border-b border-gray-100 last:border-b-0 last:pb-0"
                                                        >
                                                            <div className="font-semibold text-gray-900 leading-snug">
                                                                {item.name}
                                                            </div>
                                                            <div className="text-sm text-gray-500 mt-1">{item.category}</div>
                                                            <div className="text-xs text-gray-400 mt-1">
                                                                Закупок: {item.purchaseCount}
                                                            </div>
                                                        </div>
                                                    ))}
                                                    {frequentProducts.length === 0 && (
                                                        <div className="text-sm text-gray-500">
                                                            {isSupplier
                                                                ? 'История поставок для этого ИНН пока не найдена.'
                                                                : 'История закупок для этого ИНН пока не найдена.'}
                                                        </div>
                                                    )}
                                                    {hasHiddenFrequentProducts && (
                                                        <button
                                                            type="button"
                                                            onClick={() =>
                                                                setVisibleFrequentProductsCount((current) =>
                                                                    Math.min(current + 6, frequentProducts.length),
                                                                )
                                                            }
                                                            className="w-full rounded-[6px] border border-[#d9dde3] bg-[#f8fafc] px-3 py-2 text-sm font-semibold text-[#4b5563] transition-colors hover:border-[#c7cdd6] hover:bg-[#f1f5f9]"
                                                        >
                                                            Показать ещё
                                                        </button>
                                                    )}
                                                    {canCollapseFrequentProducts && (
                                                        <button
                                                            type="button"
                                                            onClick={() => setVisibleFrequentProductsCount(6)}
                                                            className="w-full rounded-[6px] border border-transparent px-3 py-1.5 text-sm font-medium text-[#6b7280] transition-colors hover:text-[#374151]"
                                                        >
                                                            Свернуть
                                                        </button>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Saved items button */}
                                <button
                                    type="button"
                                    onClick={() => setIsCartOpen(true)}
                                    className="relative p-2.5 text-gray-600 hover:text-[#d63d2b] transition-colors"
                                    title={savedItemsTitle}
                                >
                                    {isSupplier ? <Heart size={24} /> : <ShoppingCart size={24} />}
                                    {savedItemsCount > 0 && (
                                        <span className="absolute -top-0.5 -right-0.5 bg-[#d63d2b] text-white text-[10px] font-bold w-5 h-5 flex items-center justify-center rounded-full border-2 border-white shadow-sm">
                                            {savedItemsCount}
                                        </span>
                                    )}
                                </button>

                                {/* Logout */}
                                <button
                                    onClick={handleLogout}
                                    className="text-gray-400 hover:text-[#d63d2b] transition-colors p-2"
                                    title="Выход"
                                >
                                    <LogOut size={20} />
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center gap-4">
                                {/* Cart for guests (visible but prompt to login) */}
                                <button
                                    type="button"
                                    onClick={() => setIsCartOpen(true)}
                                    className="relative p-2.5 text-gray-600 hover:text-[#d63d2b] transition-colors"
                                    title="Корзина"
                                >
                                    <ShoppingCart size={24} />
                                    {cartProducts.length > 0 && (
                                        <span className="absolute -top-0.5 -right-0.5 bg-[#d63d2b] text-white text-[10px] font-bold w-5 h-5 flex items-center justify-center rounded-full border-2 border-white shadow-sm">
                                            {cartProducts.length}
                                        </span>
                                    )}
                                </button>

                                <button
                                    type="button"
                                    onClick={() => navigate('/login')}
                                    className="text-[16px] font-semibold text-[#6b7280] hover:text-[#404755] transition-colors"
                                >
                                    Войти
                                </button>

                                <button
                                    type="button"
                                    onClick={() => navigate('/login')}
                                    className="bg-[#d63d2b] hover:bg-[#bf3324] text-white font-bold text-[16px] px-6 py-2.5 rounded-[2px] transition-colors shadow-sm"
                                >
                                    Зарегистрироваться
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </header>

            {/* Cart slide-over */}
            <Cart isOpen={isCartOpen} onClose={() => setIsCartOpen(false)} />

            <main className="flex-grow w-full relative">
                {children}
            </main>
        </div>
    );
};

export default Layout;
