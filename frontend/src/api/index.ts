import axios from 'axios';
import {
    ApiHealthResponse,
    CartResponse,
    Item,
    OffersResponse,
    Procurement,
    SearchResponse,
    SessionEventRequest,
    SessionState,
    User,
} from '../types';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
    timeout: 15000,
});

export interface SearchEventPayload {
    userId?: string;
    inn?: string;
    region?: string;
    eventType:
        | 'search_result_click'
        | 'item_open'
        | 'item_close'
        | 'bounce'
        | 'cart_add'
        | 'cart_remove'
        | 'purchase'
        | 'item_click';
    steId?: string;
    category?: string;
    durationMs?: number;
}

export const searchProducts = async (
    query: string,
    user: User | null,
    viewedCategories: string[],
    bouncedCategories: string[],
): Promise<SearchResponse> => {
    const response = await api.post<SearchResponse>('/api/search', {
        query,
        userContext: user
            ? {
                  id: user.id,
                  inn: user.inn,
                  region: user.region,
                  viewedCategories: user.viewedCategories,
              }
            : null,
        viewedCategories,
        bouncedCategories,
    });
    return response.data;
};

export const getSuggestions = async (query: string): Promise<string[]> => {
    if (!query.trim()) {
        return [];
    }
    const response = await api.get<string[]>('/api/search/suggestions', {
        params: { q: query },
    });
    return response.data;
};

export const login = async (inn: string): Promise<User> => {
    const response = await api.post<User>('/api/auth/login', { inn });
    return response.data;
};

export const sendEvent = async (payload: SearchEventPayload): Promise<void> => {
    await api.post('/api/event', payload);
};

export const trackEvent = async (payload: SessionEventRequest): Promise<SessionState> => {
    const response = await api.post<SessionState>('/api/event', payload);
    return response.data;
};

export const getItem = async (itemId: string): Promise<Item> => {
    const response = await api.get<Item>(`/api/items/${itemId}`);
    return response.data;
};

export const getOffers = async (
    itemId: string,
    user: User | null,
): Promise<OffersResponse> => {
    const response = await api.get<OffersResponse>(`/api/items/${itemId}/offers`, {
        params: {
            user_id: user?.id ?? 'anonymous',
            customer_inn: user?.inn,
            customer_region: user?.region,
        },
    });
    return response.data;
};

export const addItemToCart = async (
    userId: string,
    steId: string,
    quantity: number = 1,
): Promise<CartResponse> => {
    const response = await api.post<CartResponse>('/api/cart/add', {
        userId,
        steId,
        quantity,
    });
    return response.data;
};

export const getCart = async (userId: string): Promise<CartResponse> => {
    const response = await api.get<CartResponse>('/api/cart', {
        params: { user_id: userId },
    });
    return response.data;
};

export const createProcurement = async (
    userId: string,
    procurementType: string = 'direct_purchase',
): Promise<Procurement> => {
    const response = await api.post<Procurement>('/api/cart/create-procurement', {
        userId,
        procurementType,
    });
    return response.data;
};

export const getHealth = async (): Promise<ApiHealthResponse> => {
    const response = await api.get<ApiHealthResponse>('/api/health');
    return response.data;
};
