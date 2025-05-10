import { Github } from "lucide-react";

const Footer = () => {
  return (
    <footer className="w-full py-6 px-4 md:px-6 border-t bg-white dark:bg-slate-950">
      <div className="container mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center text-sm text-muted-foreground">
          <div className="mb-4 md:mb-0">
            <p>© 2025 TruthShield. All rights reserved.</p>
          </div>
          <div className="flex items-center space-x-4">
            <a href="#" className="hover:text-blue-600 transition-colors">
              Terms
            </a>
            <a href="#" className="hover:text-blue-600 transition-colors">
              Privacy
            </a>
            <a href="https://github.com" 
               className="hover:text-blue-600 transition-colors flex items-center space-x-1"
               target="_blank" rel="noopener noreferrer">
              <Github size={16} />
              <span>Source</span>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
