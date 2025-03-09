import { ScrollProgress } from "./ScrollProgress";
import { useState, useEffect } from "react";

interface NavBarProps {
  isTitleVisible: boolean;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
  topRef: React.RefObject<HTMLElement | null>;
  whatIsDPRef: React.RefObject<HTMLElement | null>;
  methodsRef: React.RefObject<HTMLElement | null>;
  discussionRef: React.RefObject<HTMLElement | null>;
}

export function NavBar({
  isTitleVisible,
  scrollToSection,
  topRef,
  whatIsDPRef,
  methodsRef,
  discussionRef,
}: NavBarProps) {
  const [activeSection, setActiveSection] = useState<string>("what-is-dp");

  useEffect(() => {
    const handleScroll = () => {
      // Get main content sections
      const whatIsDPSection = document.querySelector('[data-section="what-is-dp"]');
      const methodsSection = document.querySelector('[data-section="methods"]');
      const discussionSection = document.querySelector('[data-section="discussion"]');
      
      if (!whatIsDPSection || !methodsSection || !discussionSection) return;

      // Get section positions
      const whatIsDPRect = whatIsDPSection.getBoundingClientRect();
      const methodsRect = methodsSection.getBoundingClientRect();
      const discussionRect = discussionSection.getBoundingClientRect();

      // Get viewport height
      const viewportHeight = window.innerHeight;
      const threshold = viewportHeight * 0.3; // 30% of viewport height

      // Determine active section
      if (discussionRect.top <= threshold) {
        setActiveSection("discussion");
      } else if (methodsRect.top <= threshold) {
        setActiveSection("methods");
      } else if (whatIsDPRect.top <= threshold) {
        setActiveSection("what-is-dp");
      }
    };

    const container = document.querySelector('.h-screen.overflow-y-scroll');
    if (container) {
      container.addEventListener("scroll", handleScroll);
      handleScroll(); // Initial calculation
      return () => container.removeEventListener("scroll", handleScroll);
    }
  }, []);

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-primary-white">
      <nav className="flex justify-between items-center px-12 py-4">
        <h3 
          onClick={() => scrollToSection(topRef)}
          className="text-lg font-semibold cursor-pointer hover:text-accent transition-colors"
        >
          Privacy in Practice
        </h3>
        <ul className="flex gap-8">
          <li
            onClick={() => scrollToSection(whatIsDPRef)}
            className={`transition-colors cursor-pointer ${
              activeSection === "what-is-dp" && !isTitleVisible
                ? "text-accent font-medium" 
                : "hover:text-accent"
            }`}
          >
            1. What is Differential Privacy?
          </li>
          <li
            onClick={() => scrollToSection(methodsRef)}
            className={`transition-colors cursor-pointer ${
              activeSection === "methods" 
                ? "text-accent font-medium" 
                : "hover:text-accent"
            }`}
          >
            2. How We Applied DP
          </li>
          <li
            onClick={() => scrollToSection(discussionRef)}
            className={`transition-colors cursor-pointer ${
              activeSection === "discussion" 
                ? "text-accent font-medium" 
                : "hover:text-accent"
            }`}
          >
            3. The Feasibility of Applying DP
          </li>
        </ul>
      </nav>
      <ScrollProgress />
    </div>
  );
}
