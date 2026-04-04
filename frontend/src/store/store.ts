import { create } from 'zustand';

import { CartResponse, Procurement, Product, User } from '../types';

interface StoreState {
    user: User | null;
    viewedCategories: string[];
    bouncedCategories: string[];
    productOpenTimes: Record<string, number>;

    searchQuery: string;
    results: Product[];
    isSearching: boolean;
    suggestions: string[];
    correctedQuery: string | null;
    cart: CartResponse | null;
    lastProcurement: Procurement | null;

    setUser: (user: User) => void;
    logout: () => void;
    setSearchQuery: (query: string) => void;
    setResults: (results: Product[]) => void;
    setIsSearching: (isSearching: boolean) => void;
    setSuggestions: (suggestions: string[]) => void;
    setCorrectedQuery: (query: string | null) => void;
    setCart: (cart: CartResponse | null) => void;
    setLastProcurement: (procurement: Procurement | null) => void;

    registerViewedCategory: (category: string) => void;
    markProductOpen: (productId: string, category: string) => void;
    markProductClose: (productId: string, category: string) => boolean;
}

export const useStore = create<StoreState>((set) => ({
    user: null,
    viewedCategories: [],
    bouncedCategories: [],
    productOpenTimes: {},

    searchQuery: '',
    results: [],
    isSearching: false,
    suggestions: [],
    correctedQuery: null,
    cart: null,
    lastProcurement: null,

    setUser: (user) =>
        set({
            user,
            viewedCategories: user.viewedCategories ?? [],
            bouncedCategories: [],
            productOpenTimes: {},
            cart: null,
            lastProcurement: null,
        }),

    logout: () =>
        set({
            user: null,
            viewedCategories: [],
            bouncedCategories: [],
            productOpenTimes: {},
            results: [],
            searchQuery: '',
            suggestions: [],
            correctedQuery: null,
            cart: null,
            lastProcurement: null,
        }),

    setSearchQuery: (query) => set({ searchQuery: query }),
    setResults: (results) => set({ results }),
    setIsSearching: (isSearching) => set({ isSearching }),
    setSuggestions: (suggestions) => set({ suggestions }),
    setCorrectedQuery: (query) => set({ correctedQuery: query }),
    setCart: (cart) => set({ cart }),
    setLastProcurement: (procurement) => set({ lastProcurement: procurement }),

    registerViewedCategory: (category) =>
        set((state) => ({
            viewedCategories: [...new Set([...state.viewedCategories, category])],
        })),

    markProductOpen: (productId, category) =>
        set((state) => ({
            productOpenTimes: { ...state.productOpenTimes, [productId]: Date.now() },
            viewedCategories: [...new Set([...state.viewedCategories, category])],
        })),

    markProductClose: (productId, category) => {
        let isFastBounce = false;
        set((state) => {
            const openTime = state.productOpenTimes[productId];
            const timeSpent = openTime ? Date.now() - openTime : 10_000;
            const nextOpenTimes = { ...state.productOpenTimes };
            delete nextOpenTimes[productId];
            isFastBounce = timeSpent < 3_000;

            return {
                productOpenTimes: nextOpenTimes,
                bouncedCategories: isFastBounce
                    ? [...new Set([...state.bouncedCategories, category])]
                    : state.bouncedCategories,
            };
        });
        return isFastBounce;
    },
}));
