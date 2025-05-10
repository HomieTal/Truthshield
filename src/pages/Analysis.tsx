import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { FileUp, Link, Loader2 } from "lucide-react";
import axios from 'axios';
import { toast } from "@/components/ui/sonner";

const Analysis = () => {
  const [activeTab, setActiveTab] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [urlInput, setUrlInput] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<string[]>([]);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    setIsLoading(true);
    setResults([]);

    const formData = new FormData();
    let hasValidInput = false;

    if (activeTab === 'text' && textInput.trim()) {
      formData.append('text', textInput.trim());
      hasValidInput = true;
    } else if (activeTab === 'url' && urlInput.trim()) {
      formData.append('url', urlInput.trim());
      hasValidInput = true;
    } else if (activeTab === 'image' && imageFile) {
      formData.append('image', imageFile);
      hasValidInput = true;
    }

    if (!hasValidInput) {
      toast.error('Please provide a valid input (text, URL, or image).');
      setIsLoading(false);
      return;
    }

    try {
      const response = await axios.post('http://localhost:4000/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 seconds timeout
      });

      if (response.data.success) {
        setResults(response.data.results);
        toast.success('Analysis completed successfully!');
      } else {
        toast.error(`Analysis failed: ${response.data.error}`);
      }
    } catch (error) {
      const errorMessage = axios.isAxiosError(error)
        ? error.response?.data?.error || error.message
        : 'An unexpected error occurred';
      toast.error(`Error: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-10 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Advanced Analysis</h1>
      <p className="text-lg text-muted-foreground mb-8">
        Submit content for in-depth analysis with our multiple detection models and AI cross-verification.
      </p>

      <Card>
        <CardHeader>
          <CardTitle>Submit for Analysis</CardTitle>
          <CardDescription>
            Enter news text, URL, or upload an image containing news to analyze
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3 mb-6">
              <TabsTrigger value="text">Text</TabsTrigger>
              <TabsTrigger value="url">URL</TabsTrigger>
              <TabsTrigger value="image">Image</TabsTrigger>
            </TabsList>
            <TabsContent value="text">
              <Textarea
                placeholder="Enter news text here for analysis..."
                className="min-h-[200px]"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
              />
            </TabsContent>
            <TabsContent value="url">
              <div className="flex items-center space-x-2">
                <Link className="h-5 w-5 text-gray-400" />
                <Input
                  placeholder="Enter news URL (e.g., https://example.com/news/article)"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                />
              </div>
            </TabsContent>
            <TabsContent value="image">
              <div className="grid w-full items-center gap-4">
                <label
                  htmlFor="image-upload"
                  className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <FileUp className="h-8 w-8 text-gray-400 mb-2" />
                    <p className="mb-1 text-sm text-gray-500">
                      {imageFile ? imageFile.name : 'Click to upload or drag and drop'}
                    </p>
                    <p className="text-xs text-gray-500">PNG, JPG, JPEG</p>
                  </div>
                  <input
                    id="image-upload"
                    type="file"
                    className="hidden"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleImageChange}
                  />
                </label>
              </div>
            </TabsContent>
          </Tabs>

          <div className="mt-6 flex justify-end">
            <Button onClick={handleAnalyze} disabled={isLoading}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                'Analyze Content'
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {results.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc pl-5 space-y-2">
              {results.map((result, index) => (
                <li key={index} className="text-gray-700">{result}</li>
              ))}
            </ul>
          </CardEditorialContent>
        </Card>
      )}
    </div>
  );
};

export default Analysis;