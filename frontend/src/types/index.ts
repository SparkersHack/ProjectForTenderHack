export interface PurchaseCategory {
    category: string;
    purchaseCount: number;
    totalAmount: number;
}

export interface FrequentProduct {
    steId: string;
    name: string;
    category: string;
    purchaseCount: number;
    totalAmount: number;
}

export interface User {
    id: string;
    inn: string;
    region: string;
    entityType?: string | null;
    customerName?: string | null;
    organizationTypeCode?: string | null;
    organizationTypeLabel?: string | null;
    organizationTypeSource?: string | null;
    viewedCategories: string[];
    topCategories?: PurchaseCategory[];
    frequentProducts?: FrequentProduct[];
}

export interface Product {
    id: string;
    name: string;
    category: string;
    price: number;
    offerCount: number;
    supplierInn: string;
    descriptionPreview?: string;
    reasonToShow?: string; 
}

export interface SearchResponse {
    items: Product[];
    totalCount: number;
    total_found: number;
    has_more: boolean;
    correctedQuery?: string; // Если была опечатка
}

export interface EventResponse {
    status: string;
    userId: string;
    sessionVersion: number;
    recentCategories: string[];
    clickedSteIds: string[];
    cartSteIds: string[];
    bouncedCategories: string[];
    itemCloseOutcome: 'none' | 'forgiven' | 'applied' | 'suppressed';
}

export interface AutocompleteSuggestion {
    text: string;
    type: 'product' | 'category' | 'correction' | 'query';
    reason?: string;
    score: number;
}
