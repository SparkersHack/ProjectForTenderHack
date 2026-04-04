import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import SearchPage from './pages/SearchPage';
import Layout from './components/Layout';
import ItemPage from './pages/ItemPage';
import CartPage from './pages/CartPage';

const App = () => {
    return (
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path="/login" element={<LoginPage />} />
                    <Route path="/" element={<SearchPage />} />
                    <Route path="/items/:itemId" element={<ItemPage />} />
                    <Route path="/cart" element={<CartPage />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    );
};

export default App;
