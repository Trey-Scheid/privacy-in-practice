import Link from "next/link";

interface HeroProps {
  titleRef: (node?: Element | null | undefined) => void;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
  whatIsDPRef: React.RefObject<HTMLElement | null>;
}

export function Hero({ titleRef, scrollToSection, whatIsDPRef }: HeroProps) {
  return (
    <section
      ref={titleRef}
      className="h-screen snap-start flex flex-col justify-center"
    >
      <h1 className="text-6xl font-bold mb-4">Privacy in Practice</h1>
      <h2 className="text-2xl font-semibold mb-2">
        The Feasibility of Differential Privacy for Telemetry Analysis
      </h2>
      <div className="mb-8">
        <p className="text-primary-gray">
          Tyler Kurpanek & Chris Lum & Bradley Nathanson & Trey Scheid
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
          <li className="hover:text-accent transition-colors cursor-pointer">
            2. How We Applied DP
          </li>
          <li className="hover:text-accent transition-colors cursor-pointer">
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
  );
} 