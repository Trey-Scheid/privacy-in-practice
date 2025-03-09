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
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      // Get main content sections
      const whatIsDPSection = document.querySelector(
        '[data-section="what-is-dp"]'
      );
      const methodsSection = document.querySelector('[data-section="methods"]');
      const discussionSection = document.querySelector(
        '[data-section="discussion"]'
      );

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

    const container = document.querySelector(".h-screen.overflow-y-scroll");
    if (container) {
      container.addEventListener("scroll", handleScroll);
      handleScroll(); // Initial calculation
      return () => container.removeEventListener("scroll", handleScroll);
    }
  }, []);

  const navigateTo = (ref: React.RefObject<HTMLElement | null>) => {
    scrollToSection(ref);
    setMobileMenuOpen(false); // Close mobile menu after navigating
  };

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-primary-white">
      <nav className="flex justify-between items-center px-4 md:px-12 py-4">
        <h3
          onClick={() => navigateTo(topRef)}
          className="text-lg font-semibold cursor-pointer hover:text-accent transition-colors"
        >
          Privacy in Practice
        </h3>

        {/* Mobile menu button */}
        <button
          className="md:hidden flex items-center"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            {mobileMenuOpen ? (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            ) : (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            )}
          </svg>
        </button>

        {/* Desktop navigation */}
        <ul className="hidden md:flex gap-8">
          <li
            onClick={() => navigateTo(whatIsDPRef)}
            className={`transition-colors cursor-pointer ${
              activeSection === "what-is-dp" && !isTitleVisible
                ? "text-accent font-medium"
                : "hover:text-accent"
            }`}
          >
            1. What is Differential Privacy?
          </li>
          <li
            onClick={() => navigateTo(methodsRef)}
            className={`transition-colors cursor-pointer ${
              activeSection === "methods"
                ? "text-accent font-medium"
                : "hover:text-accent"
            }`}
          >
            2. How We Applied DP
          </li>
          <li
            onClick={() => navigateTo(discussionRef)}
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

      {/* Mobile navigation menu - overlay */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-0 bg-primary-white z-40 pt-16">
          <ul className="flex flex-col gap-6 p-6">
            <li
              onClick={() => navigateTo(whatIsDPRef)}
              className={`transition-colors cursor-pointer py-2 px-4 rounded-lg ${
                activeSection === "what-is-dp" && !isTitleVisible
                  ? "bg-accent/10 text-accent font-medium"
                  : "hover:bg-primary-gray/5"
              }`}
            >
              1. What is Differential Privacy?
            </li>
            <li
              onClick={() => navigateTo(methodsRef)}
              className={`transition-colors cursor-pointer py-2 px-4 rounded-lg ${
                activeSection === "methods"
                  ? "bg-accent/10 text-accent font-medium"
                  : "hover:bg-primary-gray/5"
              }`}
            >
              2. How We Applied DP
            </li>
            <li
              onClick={() => navigateTo(discussionRef)}
              className={`transition-colors cursor-pointer py-2 px-4 rounded-lg ${
                activeSection === "discussion"
                  ? "bg-accent/10 text-accent font-medium"
                  : "hover:bg-primary-gray/5"
              }`}
            >
              3. The Feasibility of Applying DP
            </li>
          </ul>
        </div>
      )}

      <ScrollProgress />
    </div>
  );
}
