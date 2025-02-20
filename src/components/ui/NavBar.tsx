interface NavBarProps {
  isTitleVisible: boolean;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
  whatIsDPRef: React.RefObject<HTMLElement | null>;
  methodsRef: React.RefObject<HTMLElement | null>;
}

export function NavBar({
  isTitleVisible,
  scrollToSection,
  whatIsDPRef,
  methodsRef,
}: NavBarProps) {
  return (
    <div
      className={`fixed top-0 left-0 right-0 z-50 bg-primary-white border-b border-primary-gray/10 transition-all duration-300 ${
        !isTitleVisible
          ? "translate-y-0 opacity-100"
          : "translate-y-[-100%] opacity-0"
      }`}
    >
      <nav className="flex justify-between items-center px-12 py-4">
        <h3 className="text-lg font-semibold">Privacy in Practice</h3>
        <ul className="flex gap-8">
          <li
            onClick={() => scrollToSection(whatIsDPRef)}
            className="hover:text-accent transition-colors cursor-pointer"
          >
            1. What is Differential Privacy?
          </li>
          <li
            onClick={() => scrollToSection(methodsRef)}
            className="hover:text-accent transition-colors cursor-pointer"
          >
            2. How We Applied DP
          </li>
          <li className="hover:text-accent transition-colors cursor-pointer">
            3. The Feasibility of Applying DP
          </li>
        </ul>
      </nav>
    </div>
  );
}
