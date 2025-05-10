import { Link } from "react-router-dom";
import { AlertTriangle, Menu, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";

const Header = () => {
  return (
    <header className="w-full py-4 px-4 md:px-6 border-b bg-white dark:bg-slate-950 sticky top-0 z-10">
      <div className="container mx-auto flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Shield className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold heading-gradient">TruthShield</h1>
            <p className="text-xs text-muted-foreground hidden sm:block">AI-Powered Fake News Detection</p>
          </div>
        </Link>
        
        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-6">
          <Link to="/" className="text-sm font-medium hover:text-blue-600 transition-colors">
            Home
          </Link>
          <Link to="/analysis" className="text-sm font-medium hover:text-blue-600 transition-colors">
            Analysis
          </Link>
          <Link to="/about" className="text-sm font-medium hover:text-blue-600 transition-colors">
            About
          </Link>
          <div className="flex items-center space-x-1">
            <AlertTriangle size={14} className="text-amber-500" />
            <span className="text-xs text-muted-foreground">Always verify sources</span>
          </div>
        </div>
        
        {/* Mobile Navigation */}
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="md:hidden">
              <Menu className="h-5 w-5" />
              <span className="sr-only">Toggle menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="right">
            <div className="flex flex-col space-y-4 mt-8">
              <Link to="/" className="text-lg font-medium">
                Home
              </Link>
              <Link to="/analysis" className="text-lg font-medium">
                Analysis
              </Link>
              <Link to="/about" className="text-lg font-medium">
                About
              </Link>
              <div className="flex items-center space-x-1 pt-4">
                <AlertTriangle size={14} className="text-amber-500" />
                <span className="text-xs text-muted-foreground">Always verify from multiple sources</span>
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </header>
  );
};

export default Header;
