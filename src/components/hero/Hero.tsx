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
        className="h-screen snap-start flex flex-col justify-center p-12"
      >
        <h1 className="text-6xl font-bold mb-4">Privacy in Practice</h1>
        <h2 className="text-2xl font-semibold mb-2">
          The Feasibility of Differential Privacy for Telemetry Analysis
        </h2>
        <div className="mb-8">
          <p className="text-primary-gray">
            Tyler Kurpanek, Chris Lum, Bradley Nathanson, Trey Scheid (
            <button
              onClick={() => setIsContactModalOpen(true)}
              className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
            >
              contact us
            </button>
            )
          </p>
          <p className="text-primary-gray">Mentor: Yu-Xiang Wang</p>
          <p className="text-primary-gray mt-4">
            <Link
              href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
            >
              Report
            </Link>{" "}
            |{" "}
            <Link
              href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
            >
              Poster
            </Link>{" "}
            |{" "}
            <Link
              href="https://github.com/Trey-Scheid/privacy-in-practice"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent transition-colors cursor-pointer decoration-dotted underline"
            >
              Github
            </Link>
          </p>
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-2">Table of Contents</h2>
          <ul className="space-y-2">
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
            <li
              onClick={() => scrollToSection(discussionRef)}
              className="hover:text-accent transition-colors cursor-pointer"
            >
              3. The Feasibility of Applying DP
            </li>
          </ul>
        </div>
        <div className="mt-36">
          <h2 className="text-xl font-semibold mb-2 text-accent">
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
