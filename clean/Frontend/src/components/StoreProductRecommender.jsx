
import React, { useState, useEffect } from "react";
import {
  FaSearch,
  FaCheckCircle,
  FaExclamationCircle,
  FaSpinner,
  FaHeart,
  FaStar,
  FaShoppingBag,
} from "react-icons/fa";

const API_BASE_URL = "http://localhost:8000";

const StoreProductRecommender = () => {
  const [profile, setProfile] = useState("");
  const [topKStores, setTopKStores] = useState(5);
  const [topKProducts, setTopKProducts] = useState(3);
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  const [error, setError] = useState("");
  const [serverStatus, setServerStatus] = useState("unknown");

  // Check server health on component mount
  useEffect(() => {
    checkServerHealth();
  }, []);

  const checkServerHealth = async () => {
    try {
      console.log("Checking server health...");
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        // Add credentials if needed
        credentials: "same-origin",
      });

      console.log("Health check response:", response);

      if (response.ok) {
        const data = await response.json();
        console.log("Health check data:", data);
        setServerStatus(data.status || "healthy");
        return true;
      } else {
        console.log("Health check failed with status:", response.status);
        setServerStatus("offline");
        return false;
      }
    } catch (err) {
      console.log("Health check error:", err);
      setServerStatus("offline");
      return false;
    }
  };

  const getRecommendations = async () => {
    if (!profile.trim()) {
      setError("Please enter a profile description");
      return;
    }

    setLoading(true);
    setError("");
    setRecommendations(null);

    try {
      console.log("Starting recommendation request...");

      // Check server health first
      const isServerOnline = await checkServerHealth();
      if (!isServerOnline) {
        throw new Error(
          "Backend server is not responding. Make sure it's running on port 8000 and CORS is configured."
        );
      }

      console.log("Sending recommendation request...");
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Add any additional headers if needed
        },
        credentials: "same-origin",
        body: JSON.stringify({
          profile: profile.trim(),
          top_k_stores: topKStores,
          top_k_products: topKProducts,
        }),
      });

      console.log("Recommendation response:", response);

      if (!response.ok) {
        const errorText = await response.text();
        console.log("Error response text:", errorText);
        try {
          const errorData = JSON.parse(errorText);
          throw new Error(
            errorData.detail || `Server error: ${response.status}`
          );
        } catch {
          throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
      }

      const data = await response.json();
      console.log("Recommendation data:", data);
      setRecommendations(data);
    } catch (err) {
      console.log("Recommendation error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Sample profiles - make sure these are defined properly
  const sampleProfiles = [
    "A teenage girl who loves trendy fashion accessories and makeup",
    "A mother with a 3-year-old who likes colorful toys and books",
    "A college student on a tight budget looking for affordable clothing",
    "A young professional who enjoys minimalist fashion",
    // "A teenage boy who loves gaming and streetwear",
  ];

  const handleSampleProfile = (sample) => {
    console.log("Selected sample profile:", sample);
    setProfile(sample);
    setError(""); // Clear any existing errors
  };

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Header - Full Width */}
      <div className="w-full bg-white shadow-sm border-b">
        <div className="w-full max-w-none px-6 py-6">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-3 rounded-xl">
              <span className="text-2xl">🛍️</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Store & Product Recommender
              </h1>
              <p className="text-gray-600">
                Get personalized shopping recommendations powered by AI
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Full Width */}
      <div className="w-full px-6 py-8">
        {/* Server Status */}
        <div className="mb-6">
          <div className="flex items-center space-x-2 text-sm">
            {serverStatus === "healthy" && (
              <>
                <FaCheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-green-700">Server Online</span>
              </>
            )}
            {serverStatus === "offline" && (
              <>
                <FaExclamationCircle className="h-4 w-4 text-red-500" />
                <span className="text-red-700">Server Offline</span>
              </>
            )}
            {serverStatus === "unknown" && (
              <>
                <div className="h-4 w-4 rounded-full bg-gray-400"></div>
                <span className="text-gray-700">Server Status Unknown</span>
              </>
            )}
            <button
              onClick={checkServerHealth}
              className="ml-4 px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
            >
              Refresh Status
            </button>
          </div>
        </div>

        {/* Grid Layout - Full Width */}
        <div className="w-full grid grid-cols-1 xl:grid-cols-4 lg:grid-cols-3 gap-8">
          {/* Left Panel - Input Section */}
          <div className="xl:col-span-1 lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 sticky top-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                <FaSearch className="h-6 w-6 mr-2 text-purple-600" />
                Tell Us About Yourself
              </h2>

              {/* Profile Input */}
              <div className="mb-6">
                <label
                  htmlFor="profile"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Describe your preferences, age, style, and what you're looking
                  for:
                </label>
                <textarea
                  id="profile"
                  value={profile}
                  onChange={(e) => setProfile(e.target.value)}
                  placeholder="e.g., A teenage girl who loves trendy fashion accessories, enjoys experimenting with makeup, and is looking for affordable clothing options..."
                  className="w-full h-32 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none mb-6 text-gray-800"
                />
              </div>

              {/* Sample Profiles */}
              <div className="mb-6">
                <p className="text-sm font-medium text-gray-700 mb-3">
                  Quick Start Examples:
                </p>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {sampleProfiles && sampleProfiles.length > 0 ? (
                    sampleProfiles.map((sample, index) => (
                      <button
                        key={index}
                        onClick={() => handleSampleProfile(sample)}
                        className="w-full text-left p-3 text-sm bg-gray-50 hover:bg-purple-50 rounded-lg border border-gray-200 hover:border-purple-200 transition text-gray-800"
                      >
                        {sample}
                      </button>
                    ))
                  ) : (
                    <div className="text-sm text-gray-500 italic">
                      Loading examples...
                    </div>
                  )}
                </div>
              </div>

              {/* Settings */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                  Settings
                </h3>
                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Top Stores
                    </label>
                    <select
                      value={topKStores}
                      onChange={(e) => setTopKStores(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-800"
                    >
                      <option value={3}>3 stores</option>
                      <option value={5}>5 stores</option>
                      <option value={7}>7 stores</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Top Products
                    </label>
                    <select
                      value={topKProducts}
                      onChange={(e) => setTopKProducts(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-800"
                    >
                      <option value={2}>2 products</option>
                      <option value={3}>3 products</option>
                      <option value={5}>5 products</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button
                onClick={getRecommendations}
                disabled={loading || !profile.trim()}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 px-6 rounded-xl font-semibold flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl hover:from-purple-700 hover:to-pink-700 transition-all duration-200"
              >
                {loading ? (
                  <>
                    <FaSpinner className="h-5 w-5 animate-spin" />
                    <span>Getting Recommendations...</span>
                  </>
                ) : (
                  <>
                    <FaSearch className="h-5 w-5" />
                    <span>Get Recommendations</span>
                  </>
                )}
              </button>

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
                  <div className="flex items-center space-x-2">
                    <FaExclamationCircle className="h-5 w-5 text-red-500" />
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                </div>
              )}

              {/* Debug Info */}
             
            </div>
          </div>

          {/* Right Panel - Results Section - Takes up more space */}
          <div className="xl:col-span-3 lg:col-span-2">
            {loading && (
              <div className="w-full bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
                <div className="flex flex-col items-center justify-center py-16">
                  <FaSpinner className="h-16 w-16 text-purple-600 animate-spin mb-6" />
                  <p className="text-gray-600 text-xl mb-2">
                    Analyzing your preferences...
                  </p>
                  <p className="text-gray-500 text-sm">
                    This may take a few seconds
                  </p>
                </div>
              </div>
            )}

            {recommendations && !loading && (
              <div className="w-full space-y-6">
                {/* Store Recommendations */}{" "}
                <div className="w-full bg-white rounded-2xl shadow-xl p-8 border border-gray-100 hover:shadow-2xl transition-all duration-300">
                  {" "}
                  <h2 className="text-3xl font-bold text-gray-900 mb-4 flex items-center">
                    {/* <FaShoppingBag className="h-8 w-8 mr-3 text-yellow-500" /> */}
                    Recommended Stores{" "}
                  </h2>
                  {/* ← NEW OVERVIEW HEADING */}{" "}
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">
                    Overview
                  </h3>{" "}
                  <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                    {" "}
                    <pre className="text-base text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
                      {recommendations.stores_prompt}{" "}
                    </pre>{" "}
                  </div>{" "}
                </div>
                {/* Product Recommendations */}
                {Object.entries(recommendations.product_prompts).map(
                  ([store, products]) => (
                    <div
                      key={store}
                      className="w-full bg-white rounded-2xl shadow-xl p-8 border border-gray-100 hover:shadow-2xl transition-all duration-300"
                    >
                      <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                        {/* ← HEARTS → SHOPPING BAG */}{" "}
                        <FaShoppingBag className="h-6 w-6 mr-3 text-gray-500" />
                        Products from{" "}
                        {store.charAt(0).toUpperCase() + store.slice(1)}+{" "}
                      </h3>
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-100">
                        <pre className="text-base text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
                          {products}
                        </pre>
                      </div>
                    </div>
                  )
                )}
              </div>
            )}

            {!recommendations && !loading && (
              <div className="w-full bg-white rounded-2xl shadow-xl p-12 border border-gray-100">
                <div className="text-center py-16">
                  <FaShoppingBag className="h-24 w-24 text-gray-300 mx-auto mb-6" />
                  <h3 className="text-2xl font-semibold text-gray-900 mb-4">
                    Ready to Find Your Perfect Stores?
                  </h3>
                  <p className="text-gray-600 mb-8 text-lg">
                    Enter your preferences in the form to get personalized store
                    and product recommendations.
                  </p>
                  <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-8 border border-purple-100 max-w-2xl mx-auto">
                    <h4 className="font-semibold text-gray-800 mb-4 text-lg">
                      How it works:
                    </h4>
                    <div className="text-left space-y-3 text-base text-gray-700">
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-purple-500 rounded-full flex-shrink-0" />
                        <span>
                          Describe your style, age, and shopping preferences
                        </span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-pink-500 rounded-full flex-shrink-0" />
                        <span>
                          Our AI analyzes your profile against store data
                        </span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="w-3 h-3 bg-purple-500 rounded-full flex-shrink-0" />
                        <span>
                          Get personalized store and product recommendations
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer - Full Width */}
        <div className="w-full mt-16 text-center">
          <p className="text-gray-500 text-sm">
            Powered by AI • Vector Search • Machine Learning
          </p>
        </div>
      </div>
    </div>
  );
};

export default StoreProductRecommender;
