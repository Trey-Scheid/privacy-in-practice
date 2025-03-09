import Link from "next/link";
import { useState } from "react";
import { ContactModal } from "./ContactModal";

interface HeroProps {
  titleRef: (node?: Element | null | undefined) => void;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
  whatIsDPRef: React.RefObject<HTMLElement>;
  methodsRef: React.RefObject<HTMLElement>;
  discussionRef: React.RefObject<HTMLElement>;
}

export function Hero({
  titleRef,
  scrollToSection,
  whatIsDPRef,
  methodsRef,
  discussionRef,
}: HeroProps) {
  const [isContactModalOpen, setIsContactModalOpen] = useState(false);

  return (
    <>
      <section
        ref={titleRef}
        className="h-screen snap-start flex flex-col justify-between p-6 md:p-12"
      >
        <div className="mt-16 md:mt-0">
          <h1 className="text-4xl md:text-6xl font-bold mb-2 md:mb-4">
            Privacy in Practice
          </h1>
          <h2 className="text-xl md:text-2xl font-semibold mb-2">
            The Feasibility of Differential Privacy for Telemetry Analysis
          </h2>
          <div className="mb-6 md:mb-8">
            <p className="text-primary-gray text-sm md:text-base">
              Tyler Kurpanek, Chris Lum, Bradley Nathanson, Trey Scheid (
              <button
                onClick={() => setIsContactModalOpen(true)}
                className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
              >
                contact us
              </button>
              )
            </p>
            <p className="text-primary-gray text-sm md:text-base">
              Mentor: Yu-Xiang Wang
            </p>
            <p className="text-primary-gray text-sm md:text-base mt-3 md:mt-4">
              <Link
                href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
              >
                Report
              </Link>
              {" | "}
              <Link
                href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
              >
                Poster
              </Link>
              {" | "}
              <Link
                href="https://github.com/Trey-Scheid/privacy-in-practice"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
              >
                Github
              </Link>
              {" | "}
              <Link
                href="https://endurable-gatsby-6d6.notion.site/Privacy-In-Practice-14556404e74780818747cbe76de2e04a"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
              >
                Notion
              </Link>
            </p>
          </div>
          <div>
            <h2 className="text-lg md:text-xl font-semibold mb-2">
              Table of Contents
            </h2>
            <ul className="space-y-2">
              <li
                onClick={() => scrollToSection(whatIsDPRef)}
                className="text-sm md:text-base hover:text-accent transition-colors cursor-pointer"
              >
                1. What is Differential Privacy?
              </li>
              <li
                onClick={() => scrollToSection(methodsRef)}
                className="text-sm md:text-base hover:text-accent transition-colors cursor-pointer"
              >
                2. How We Applied DP
              </li>
              <li
                onClick={() => scrollToSection(discussionRef)}
                className="text-sm md:text-base hover:text-accent transition-colors cursor-pointer"
              >
                3. The Feasibility of Applying DP
              </li>
            </ul>
          </div>
        </div>

        <div className="mb-8 md:mt-36 self-center text-center">
          <h2 className="text-lg md:text-xl font-semibold text-accent animate-bounce">
            ↓ What is Differential Privacy? ↓
          </h2>
        </div>
      </section>

      <ContactModal
        isOpen={isContactModalOpen}
        onClose={() => setIsContactModalOpen(false)}
      />
    </>
  );
}
