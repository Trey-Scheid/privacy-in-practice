import { useEffect, useState } from "react";

interface Section {
  color: string;
  lightColor: string;
  subsections?: number;
  name: string; // Add name for mobile labels
}

const sections: Section[] = [
  {
    color: "rgb(148, 163, 184)", // slate-400
    lightColor: "rgb(195, 201, 208)", // slate-200
    name: "What is DP",
  },
  {
    color: "rgb(118, 171, 174)", // accent
    lightColor: "rgb(191, 216, 218)", // accent light
    subsections: 4,
    name: "Our Methods",
  },
  {
    color: "rgb(44, 51, 51)", // primary-gray
    lightColor: "rgb(164, 164, 164)", // primary-gray light
    name: "Discussion",
  },
];

export function ScrollProgress() {
  const [sectionProgress, setSectionProgress] = useState<number[]>([0, 0, 0]);
  const [activeSectionIndex, setActiveSectionIndex] = useState<number>(0);

  useEffect(() => {
    const handleScroll = () => {
      // Get the main scrollable container
      const container = document.querySelector(".h-screen.overflow-y-auto");
      if (!container) return;

      // Get main content sections
      const whatIsDPSection = document.querySelector(
        '[data-section="what-is-dp"]'
      );
      const methodsSection = document.querySelector('[data-section="methods"]');
      const discussionSection = document.querySelector(
        '[data-section="discussion"]'
      );

      if (!whatIsDPSection || !methodsSection || !discussionSection) return;

      // Get section positions relative to the container
      const whatIsDPRect = whatIsDPSection.getBoundingClientRect();
      const methodsRect = methodsSection.getBoundingClientRect();
      const discussionRect = discussionSection.getBoundingClientRect();

      // Get container dimensions
      const containerRect = container.getBoundingClientRect();
      const viewportHeight = containerRect.height;
      const threshold = viewportHeight * 0.3; // 30% of viewport height

      // Calculate progress for each section
      const newProgress = [0, 0, 0];

      // What is DP section progress
      const whatIsDPProgress = Math.min(
        100,
        Math.max(
          0,
          ((viewportHeight - whatIsDPRect.top) / whatIsDPRect.height) * 100
        )
      );
      newProgress[0] = whatIsDPProgress;

      // Methods section progress
      const methodsProgress = Math.min(
        100,
        Math.max(
          0,
          ((viewportHeight - methodsRect.top) / methodsRect.height) * 100
        )
      );
      newProgress[1] = methodsProgress;

      // Discussion section progress
      const discussionProgress = Math.min(
        100,
        Math.max(
          0,
          ((viewportHeight - discussionRect.top) / discussionRect.height) * 100
        )
      );
      newProgress[2] = discussionProgress;

      setSectionProgress(newProgress);

      // Determine active section
      if (discussionRect.top <= threshold) {
        setActiveSectionIndex(2);
      } else if (methodsRect.top <= threshold) {
        setActiveSectionIndex(1);
      } else {
        setActiveSectionIndex(0);
      }
    };

    // Get the main scrollable container
    const container = document.querySelector(".h-screen.overflow-y-auto");
    if (container) {
      container.addEventListener("scroll", handleScroll);
      handleScroll(); // Initial calculation
      return () => container.removeEventListener("scroll", handleScroll);
    }
  }, []);

  return (
    <>
      {/* Mobile section indicator - only visible on small screens */}
      <div className="md:hidden text-xs font-medium text-center py-1 bg-primary-gray/5">
        {sections[activeSectionIndex].name}
      </div>

      {/* Progress bar */}
      <div className="h-1.5 md:h-1 w-full flex">
        {sections.map((section, i) => {
          const progress = Math.min(100, Math.max(0, sectionProgress[i]));

          return (
            <div
              key={i}
              className="h-full relative overflow-hidden"
              style={{
                width: `${100 / sections.length}%`,
                backgroundColor: section.lightColor,
              }}
            >
              {/* Dark progress bar */}
              <div
                className="absolute top-0 left-0 h-full transition-all duration-200"
                style={{
                  width: `${progress}%`,
                  backgroundColor: section.color,
                }}
              />

              {/* Subsection dividers */}
              {section.subsections && (
                <div className="absolute top-0 left-0 w-full h-full flex">
                  {Array.from({ length: section.subsections || 1 }, (_, j) => {
                    const subsectionWidth = 100 / (section.subsections || 1);
                    return (
                      <div
                        key={j}
                        className="h-full border-r last:border-r-0 border-white/10"
                        style={{ width: `${subsectionWidth}%` }}
                      />
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}
