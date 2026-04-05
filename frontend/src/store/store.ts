import { create } from 'zustand';
import { User, Product, AutocompleteSuggestion, SearchResponse } from '../types';
import { searchProducts, sendEvent } from '../api';

interface StoreState {
  // User state
  user: User | null;
  setUser: (user: User) => void;
  logout: () => void;
  // User history and behavioral
  viewedCategories: string[]; 
  bouncedCategories: string[]; // Penalized categories
  productOpenTimes: Record<string, number>; // productId -> timestamp of opening
  cartCloseGraceProductIds: Record<string, boolean>;
  cartProducts: Product[];
  favoriteProducts: Product[];

  // Search Context
  searchQuery: string;
  results: Product[];
  isSearching: boolean;
  suggestions: AutocompleteSuggestion[];
  correctedQuery: string | null;
  totalFound: number;
  hasMore: boolean;
  searchOffset: number;
  searchLimit: number;
  minScore: number;

  setSearchQuery: (query: string) => void;
  setIsSearching: (isSearching: boolean) => void;
  setSuggestions: (suggestions: AutocompleteSuggestion[]) => void;
  setCorrectedQuery: (query: string | null) => void;
  setSearchResponse: (response: SearchResponse, append?: boolean) => void;
  resetSearchResults: () => void;

  // Actions
  trackProductParams: (category: string) => void;
  simulateProductOpen: (productId: string, category: string) => void;
  simulateProductClose: (productId: string, category: string) => void;
  addToCart: (product: Product) => void;
  removeFromCart: (productId: string) => void;
  clearCart: () => void;
  addToFavorites: (product: Product) => void;
  removeFromFavorites: (productId: string) => void;
  clearFavorites: () => void;
}

const refreshCurrentSearch = async () => {
  const {
    searchQuery,
    user,
    viewedCategories,
    bouncedCategories,
    searchLimit,
    minScore,
    setSearchResponse,
    setCorrectedQuery,
    setIsSearching,
  } = useStore.getState();
  const refreshQuery = searchQuery.trim();
  if (!refreshQuery) {
    return;
  }

  setIsSearching(true);
  try {
    const refreshedResponse = await searchProducts(
      refreshQuery,
      user,
      viewedCategories,
      bouncedCategories,
      {
        limit: searchLimit,
        offset: 0,
        minScore,
      },
    );
    if (useStore.getState().searchQuery.trim() !== refreshQuery) {
      return;
    }
    setSearchResponse(refreshedResponse);
    setCorrectedQuery(refreshedResponse.correctedQuery || null);
  } finally {
    if (useStore.getState().searchQuery.trim() === refreshQuery) {
      setIsSearching(false);
    }
  }
};

export const useStore = create<StoreState>((set, get) => ({
  user: null,
  setUser: (user) =>
    set({
      user,
      viewedCategories: user.viewedCategories ?? [],
      bouncedCategories: [],
      productOpenTimes: {},
      cartCloseGraceProductIds: {},
      cartProducts: [],
      favoriteProducts: [],
    }),
  logout: () =>
    set({
      user: null,
      viewedCategories: [],
      bouncedCategories: [],
      productOpenTimes: {},
      cartCloseGraceProductIds: {},
      cartProducts: [],
      favoriteProducts: [],
      results: [],
      searchQuery: '',
      suggestions: [],
      correctedQuery: null,
      totalFound: 0,
      hasMore: false,
      searchOffset: 0,
    }),

  viewedCategories: [],
  bouncedCategories: [],
  productOpenTimes: {},
  cartCloseGraceProductIds: {},
  cartProducts: [],
  favoriteProducts: [],

  searchQuery: '',
  results: [],
  isSearching: false,
  suggestions: [],
  correctedQuery: null,
  totalFound: 0,
  hasMore: false,
  searchOffset: 0,
  searchLimit: 20,
  minScore: 0.55,

  setSearchQuery: (query) => set({ searchQuery: query }),
  setIsSearching: (isSearching) => set({ isSearching }),
  setSuggestions: (suggestions) => set({ suggestions }),
  setCorrectedQuery: (query) => set({ correctedQuery: query }),
  setSearchResponse: (response, append = false) =>
    set((state) => {
      const nextResults = append ? [...state.results, ...response.items] : response.items;
      const totalFound = response.total_found ?? response.totalCount ?? nextResults.length;
      return {
        results: nextResults,
        totalFound,
        hasMore: response.has_more,
        searchOffset: nextResults.length,
      };
    }),
  resetSearchResults: () =>
    set({
      results: [],
      totalFound: 0,
      hasMore: false,
      searchOffset: 0,
    }),

  trackProductParams: (category) =>
    set((state) => ({
      viewedCategories: [...new Set([...state.viewedCategories, category])]
    })),

  simulateProductOpen: (productId, category) => {
    const { user } = get();
    set((state) => ({
      productOpenTimes: { ...state.productOpenTimes, [productId]: Date.now() },
    }));
    void sendEvent({
      userId: user?.id,
      inn: user?.inn,
      region: user?.region,
      eventType: 'item_open',
      steId: productId,
      category,
    })
      .then((response) => {
        set({
          viewedCategories: [...new Set((response.recentCategories ?? []).filter((value) => value && value.trim().length > 0))],
        });
      })
      .catch((error) => console.error(error));
  },

  simulateProductClose: (productId, category) => {
    const { user, productOpenTimes, cartCloseGraceProductIds } = get();
    const openTime = productOpenTimes[productId];
    const timeSpent = openTime ? Date.now() - openTime : 10000;
    const closeReason = cartCloseGraceProductIds[productId] ? 'after_cart_add' : undefined;

    set((state) => {
      const nextOpenTimes = { ...state.productOpenTimes };
      const nextCartCloseGraceProductIds = { ...state.cartCloseGraceProductIds };
      delete nextOpenTimes[productId];
      delete nextCartCloseGraceProductIds[productId];
      return {
        productOpenTimes: nextOpenTimes,
        cartCloseGraceProductIds: nextCartCloseGraceProductIds,
        bouncedCategories: state.bouncedCategories,
      };
    });

    void (async () => {
      try {
        const response = await sendEvent({
          userId: user?.id,
          inn: user?.inn,
          region: user?.region,
          eventType: 'item_close',
          steId: productId,
          category,
          durationMs: timeSpent,
          closeReason,
        });
        const nextBouncedCategories = [
          ...new Set((response.bouncedCategories ?? []).filter((value) => value && value.trim().length > 0)),
        ];
        set({
          viewedCategories: [...new Set((response.recentCategories ?? []).filter((value) => value && value.trim().length > 0))],
          bouncedCategories: nextBouncedCategories,
        });

        if (response.itemCloseOutcome !== 'applied') {
          return;
        }

        const {
          searchQuery,
          user: activeUser,
          viewedCategories,
          searchLimit,
          minScore,
          setSearchResponse,
          setCorrectedQuery,
          setIsSearching,
        } = get();
        const refreshQuery = searchQuery.trim();
        if (!refreshQuery) {
          return;
        }

        setIsSearching(true);
        try {
          const refreshedResponse = await searchProducts(
            refreshQuery,
            activeUser,
            viewedCategories,
            nextBouncedCategories,
            {
              limit: searchLimit,
              offset: 0,
              minScore,
            },
          );
          if (get().searchQuery.trim() !== refreshQuery) {
            return;
          }
          setSearchResponse(refreshedResponse);
          setCorrectedQuery(refreshedResponse.correctedQuery || null);
        } finally {
          if (get().searchQuery.trim() === refreshQuery) {
            setIsSearching(false);
          }
        }
      } catch (error) {
        console.error(error);
      }
    })();
  },

  addToCart: (product) => {
    const { user } = get();
    set((state) => ({
      cartProducts: state.cartProducts.some((item) => item.id === product.id)
        ? state.cartProducts
        : [...state.cartProducts, product],
      cartCloseGraceProductIds: state.productOpenTimes[product.id]
        ? { ...state.cartCloseGraceProductIds, [product.id]: true }
        : state.cartCloseGraceProductIds,
    }));
    void (async () => {
      try {
        const response = await sendEvent({
          userId: user?.id,
          inn: user?.inn,
          region: user?.region,
          eventType: 'cart_add',
          steId: product.id,
          category: product.category,
        });
        set({
          viewedCategories: [...new Set((response.recentCategories ?? []).filter((value) => value && value.trim().length > 0))],
          bouncedCategories: [...new Set((response.bouncedCategories ?? []).filter((value) => value && value.trim().length > 0))],
        });
        await refreshCurrentSearch();
      } catch (error) {
        console.error(error);
      }
    })();
  },

  removeFromCart: (productId) => {
    const { user, cartProducts } = get();
    const productToRemove = cartProducts.find((item) => item.id === productId);
    set((state) => ({
      cartProducts: state.cartProducts.filter((item) => item.id !== productId),
    }));
    if (productToRemove) {
      void sendEvent({
        userId: user?.id,
        inn: user?.inn,
        region: user?.region,
        eventType: 'cart_remove',
        steId: productId,
        category: productToRemove.category,
      }).catch((error) => console.error(error));
    }
  },

  clearCart: () => {
    const { user, cartProducts } = get();
    cartProducts.forEach((product) => {
      void sendEvent({
        userId: user?.id,
        inn: user?.inn,
        region: user?.region,
        eventType: 'purchase',
        steId: product.id,
        category: product.category,
      }).catch((error) => console.error(error));
    });
    set({ cartProducts: [] });
  },

  addToFavorites: (product) =>
    set((state) => ({
      favoriteProducts: state.favoriteProducts.some((item) => item.id === product.id)
        ? state.favoriteProducts
        : [...state.favoriteProducts, product],
    })),

  removeFromFavorites: (productId) =>
    set((state) => ({
      favoriteProducts: state.favoriteProducts.filter((item) => item.id !== productId),
    })),

  clearFavorites: () => set({ favoriteProducts: [] }),
}));
