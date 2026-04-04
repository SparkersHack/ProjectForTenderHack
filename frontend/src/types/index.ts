export interface User {
    id: string;
    inn: string;
    region: string;
    viewedCategories: string[];
}

export interface Product {
    id: string;
    name: string;
    category: string;
    price: number;
    supplierInn: string;
    descriptionPreview?: string;
    reasonToShow?: string;
}

export interface SearchResponse {
    items: Product[];
    totalCount: number;
    correctedQuery?: string | null;
}

export interface SessionEventRequest {
    userId: string;
    eventType: string;
    steId?: string;
    category?: string;
    query?: string;
    metadata?: Record<string, unknown>;
}

export interface SessionState {
    userId: string;
    sessionVersion?: number;
    clickedSteIds: string[];
    cartSteIds: string[];
    recentCategories: string[];
    viewedCategories: string[];
    bouncedCategories: string[];
    eventCount: number;
}

export interface Item {
    id: string;
    name: string;
    category: string;
    normalizedCategory: string;
    attributeKeys: string[];
    attributeCount: number;
    keyTokens: string[];
    price: number;
    supplierInn: string;
    supplierRegion: string;
    offerCount: number;
}

export interface Offer {
    offerId: string;
    steId: string;
    supplierInn: string;
    supplierRegion: string;
    unitPrice: number;
    offerScore: number;
    explanation: string[];
}

export interface OffersResponse {
    itemId: string;
    offers: Offer[];
}

export interface CartItem {
    steId: string;
    name: string;
    category: string;
    quantity: number;
    price: number;
    supplierInn: string;
    addedAt: string;
}

export interface CartResponse {
    userId: string;
    items: CartItem[];
    totalItems: number;
    totalAmount: number;
}

export interface Procurement {
    procurementId: string;
    procurementType: string;
    status: string;
    itemCount: number;
    totalAmount: number;
    createdAt: string;
}

export interface HealthCheck {
    path: string;
    exists: boolean;
}

export interface ApiHealthResponse {
    status: string;
    service: string;
    error?: string | null;
    assets: {
        ready: boolean;
        required: Record<string, HealthCheck>;
        optional: Record<string, HealthCheck>;
    };
}
