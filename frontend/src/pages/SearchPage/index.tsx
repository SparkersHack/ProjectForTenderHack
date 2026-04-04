import React from 'react';
import SearchBar from '../../components/SearchBar/SearchBar';
import ProductCard from '../../components/ProductCard';
import { useStore } from '../../store/store';
import { Loader2 } from 'lucide-react';

const SearchPage = () => {
    const { results, isSearching, searchQuery, correctedQuery } = useStore();

    const hasNoResults = results.length === 0 && searchQuery.length > 0 && !isSearching;
    const isPristine = results.length === 0 && searchQuery.length === 0 && !isSearching && !correctedQuery;

    return (
        <div className="w-full max-w-[1400px] mx-auto px-6 pt-10 pb-20">
            <div className="flex flex-col items-center mb-20 md:mb-24 mt-6">
                <h1 className="text-4xl md:text-[44px] font-bold text-gray-900 mb-10 text-center tracking-tight">
                    Единый каталог продукции
                </h1>
                
                <div className="w-full relative z-30 px-4 md:px-0">
                    <SearchBar />
                </div>
            </div>

            <div className="w-full max-w-[1400px] mx-auto relative z-0">
                {isSearching && (
                    <div className="flex flex-col items-center justify-center py-32 text-gray-500">
                        <Loader2 className="animate-spin mb-6 text-[#E03F3F]" size={48} />
                        <p className="text-lg font-medium">Интеллектуальный поиск по каталогу...</p>
                    </div>
                )}

                {!isSearching && isPristine && (
                    <div className="text-center py-32 text-gray-500">
                        <p className="text-xl font-medium text-gray-700">Начните поиск, введя название товара или категорию.</p>
                        <p className="mt-3 text-base text-gray-400">Например: "Бумага А4 Снегурочка"</p>
                    </div>
                )}

                {!isSearching && hasNoResults && (
                    <div className="text-center py-32 text-gray-600 bg-white rounded-[8px] border border-gray-200 shadow-sm">
                        <p className="text-2xl font-bold mb-3 text-gray-900">Ничего не найдено</p>
                        <p className="text-lg text-gray-500">Убедитесь, что запрос написан без ошибок, или попробуйте использовать другие синонимы.</p>
                    </div>
                )}

                {!isSearching && results.length > 0 && (
                    <>
                        <div className="mb-6 flex items-center justify-between text-sm text-gray-500">
                            <div>Найдено позиций: <span className="font-semibold text-gray-900">{results.length}</span></div>
                            {correctedQuery && (
                                <div className="hidden md:block">
                                    Исправленный запрос: <span className="font-semibold text-gray-900">{correctedQuery}</span>
                                </div>
                            )}
                        </div>
                        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                            {results.map((product) => (
                                <ProductCard key={product.id} product={product} />
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default SearchPage;
