import axios from 'axios';
import { AutocompleteSuggestion, EventResponse, SearchResponse, User } from '../types';

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
    closeReason?: 'dismiss' | 'after_cart_add';
}

export interface SearchProductsOptions {
    limit?: number;
    offset?: number;
    minScore?: number;
}

export const searchProducts = async (
    query: string,
    user: User | null,
    viewedCategories: string[],
    bouncedCategories: string[],
    options: SearchProductsOptions = {},
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
        limit: options.limit,
        offset: options.offset ?? 0,
        min_score: options.minScore ?? 0.55,
    });
    return response.data;
};

export const getSuggestions = async (
    query: string,
    user: User | null,
    viewedCategories: string[],
): Promise<AutocompleteSuggestion[]> => {
    if (!query.trim()) {
        return [];
    }

    const params: Record<string, string> = { q: query, top_k: '8' };
    const mergedViewedCategories = [...new Set([...(user?.viewedCategories ?? []), ...viewedCategories])]
        .filter((value) => value && value.trim().length > 0);
    const topCategories = (user?.topCategories ?? [])
        .map((item) => item.category)
        .filter((value) => value && value.trim().length > 0);

    if (user?.inn) {
        params.inn = user.inn;
    }
    if (mergedViewedCategories.length) {
        params.viewed_categories = mergedViewedCategories.join("|");
    }
    if (topCategories.length) {
        params.top_categories = topCategories.join("|");
    }

    const response = await api.get<AutocompleteSuggestion[]>('/api/search/suggestions', {
        params,
    });
    return response.data;
};

export const login = async (inn: string): Promise<User> => {
    const response = await api.post<User>('/api/auth/login', { inn });
    return response.data;
};

export const sendEvent = async (payload: SearchEventPayload): Promise<EventResponse> => {
    const response = await api.post<EventResponse>('/api/event', payload);
    return response.data;
};
