import React from 'react';

const About = () => {
  return (
    <div className="container mx-auto px-4 py-10 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">About TruthShield</h1>
      <div className="prose max-w-none">
        <p className="text-lg mb-4">
          TruthShield is an AI-powered fake news detection platform designed to help users identify potentially misleading information.
        </p>
        
        <h2 className="text-2xl font-semibold mt-8 mb-4">Our Technology</h2>
        <p className="mb-4">
          Our system uses multiple machine learning models and natural language processing techniques to analyze news content:
        </p>
        <ul className="list-disc pl-6 mb-6">
          <li className="mb-2">SGD Classifier - A supervised learning algorithm trained on thousands of verified news samples</li>
          <li className="mb-2">Passive Aggressive Classifier - A specialized algorithm for text classification tasks</li>
          <li className="mb-2">AI Cross-checking - Advanced language models analyze content for journalistic standards</li>
        </ul>
        
        <h2 className="text-2xl font-semibold mt-8 mb-4">How It Works</h2>
        <p className="mb-4">
          Simply input news text, a URL, or upload an image containing news content. Our system will analyze the content and provide an assessment of its credibility along with confidence scores.
        </p>
        
        <div className="bg-amber-50 border-l-4 border-amber-500 p-4 my-6">
          <p className="text-amber-700">
            <strong>Important:</strong> No automated system is perfect. Always cross-check information from multiple reliable sources.
          </p>
        </div>
      </div>
    </div>
  );
};

export default About;