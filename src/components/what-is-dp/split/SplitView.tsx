import { useEffect, useState, useRef } from "react";
import { useInView } from "react-intersection-observer";
import Link from "next/link";
import { InlineMath, BlockMath } from "react-katex";
import { MonaLisa } from "./MonaLisa";
import { Hero } from "@/components/hero/Hero";
import { getPublicPath } from "@/lib/utils";

interface SplitViewProps {
  SplitViewRef: React.RefObject<HTMLElement>;
  titleRef: (node?: Element | null | undefined) => void;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
  methodsRef: React.RefObject<HTMLElement>;
  discussionRef: React.RefObject<HTMLElement>;
}

export function SplitView({
  SplitViewRef,
  titleRef,
  scrollToSection,
  methodsRef,
  discussionRef,
}: SplitViewProps) {
  const [monaLisa, setMonaLisa] = useState("");
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [selectedEpsilon, setSelectedEpsilon] = useState<number | null>(null);
  const { ref: addNoiseML, inView: pastNoiseTrigger } = useInView({
    threshold: 0.5,
    triggerOnce: true,
  });

  // Specific ref for the mobile noise section
  const { ref: mobileTriggerRef, inView: mobileNoiseTrigger } = useInView({
    threshold: 0.3,
    triggerOnce: true,
  });

  // Track if we're in the split view section
  const splitViewContainerRef = useRef<HTMLDivElement>(null);
  const [isInSplitView, setIsInSplitView] = useState(true);
  const mobileLisaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsInSplitView(entry.isIntersecting);
      },
      {
        threshold: 0,
        rootMargin: "0px 0px -90% 0px", // Make it stay visible longer
      }
    );

    if (splitViewContainerRef.current) {
      observer.observe(splitViewContainerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    fetch(getPublicPath("/monalisa.txt"))
      .then((response) => response.text())
      .then((content) => setMonaLisa(content))
      .catch((error) => {
        console.error("Failed to load Mona Lisa:", error);
        setMonaLisa("");
      });
  }, []);

  // Convert epsilon to probability
  const epsilonToProbability = (epsilon: number) => {
    return 1 / (1 + Math.exp(epsilon));
  };

  // Handle epsilon button click
  const handleEpsilonClick = (epsilon: number) => {
    setSelectedEpsilon(epsilon);
    const targetNoise = epsilonToProbability(epsilon);

    // Animate to new noise level
    let currentNoise = noiseLevel;
    const duration = 1000; // 1 second
    const steps = 20;
    const increment = (targetNoise - currentNoise) / steps;
    const stepDuration = duration / steps;

    const timer = setInterval(() => {
      if (Math.abs(currentNoise - targetNoise) > Math.abs(increment)) {
        currentNoise += increment;
        setNoiseLevel(currentNoise);
      } else {
        setNoiseLevel(targetNoise);
        clearInterval(timer);
      }
    }, stepDuration);
  };

  // Animate noise level when section comes into view
  useEffect(() => {
    if (pastNoiseTrigger || mobileNoiseTrigger) {
      let currentNoise = 0;
      const targetNoise = 0.2;
      const duration = 1500; // 1.5 seconds
      const steps = 20; // Number of steps in the animation
      const increment = targetNoise / steps;
      const stepDuration = duration / steps;

      const timer = setInterval(() => {
        currentNoise += increment;
        setNoiseLevel(currentNoise);
        if (currentNoise >= targetNoise) {
          clearInterval(timer);
        }
      }, stepDuration);

      return () => clearInterval(timer);
    }
  }, [pastNoiseTrigger, mobileNoiseTrigger]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNoiseLevel(parseFloat(e.target.value));
  };

  return (
    <>
      {/* Mobile layout - Vertical stack with Mona Lisa as background */}
      <div className="md:hidden relative" ref={splitViewContainerRef}>
        {/* Fixed Mona Lisa Background - stays in view while in SplitView section */}
        <div
          ref={mobileLisaRef}
          className="fixed inset-0 z-0 flex items-center justify-center bg-primary-white"
          style={{
            opacity: isInSplitView ? 1 : 0,
            transition: "opacity 0.3s ease-in-out",
            pointerEvents: "none", // Let clicks go through to content
          }}
        >
          <div className="transform scale-90 opacity-30">
            <MonaLisa
              ascii={monaLisa}
              shouldAddNoise={mobileNoiseTrigger} // Only add noise when triggered by scrolling
              noiseLevel={noiseLevel}
              showOriginal={showOriginal}
            />
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="relative z-10">
          {/* Title Section */}
          <section
            ref={titleRef as React.LegacyRef<HTMLElement>}
            className="min-h-screen flex flex-col justify-center p-6"
          >
            <Hero
              titleRef={titleRef}
              scrollToSection={scrollToSection}
              whatIsDPRef={SplitViewRef}
              methodsRef={methodsRef}
              discussionRef={discussionRef}
            />
          </section>

          {/* What is DP Section */}
          <section
            ref={SplitViewRef as React.LegacyRef<HTMLElement>}
            className="min-h-[80vh] flex flex-col justify-center p-6"
          >
            <div className="prose prose-lg max-w-none bg-primary-white/80 rounded-lg p-6 shadow-md">
              <h1 className="text-3xl font-bold mb-4">
                What is Differential Privacy?
              </h1>
              <p className="text-lg">
                In today&apos;s data-driven world, the need to protect
                individual privacy while maintaining the utility of data
                analysis has become increasingly crucial. Differential Privacy
                (DP) emerges as a mathematical framework that provides strong
                privacy guarantees while allowing meaningful statistical
                analysis
              </p>
            </div>
          </section>

          {/* Core Concept Section */}
          <section className="min-h-[80vh] flex flex-col justify-center p-6">
            <div className="prose prose-lg max-w-none bg-primary-white/80 rounded-lg p-6 shadow-md">
              <p className="text-lg">
                At its core, differential privacy ensures that the presence or
                absence of any individual&apos;s data in a dataset does not
                significantly affect the results of any analysis performed on
                that dataset. This is achieved by carefully introducing random
                noise into the computation process, making it virtually
                impossible to reverse-engineer individual records...
              </p>
            </div>
          </section>

          {/* Randomized Response Section */}
          <section
            ref={mobileTriggerRef}
            className="min-h-[80vh] flex flex-col justify-center p-6"
          >
            <div className="prose prose-lg max-w-none bg-primary-white/80 rounded-lg p-6 shadow-md">
              <p className="text-lg">
                ...all while maintaining the big picture!
              </p>
              <p className="text-lg mt-4">
                But what just happened? We added noise to the image of Mona Lisa
                by probabilistically flipping each pixel. This way, you can
                still see the big picture, but the individual pixels&apos; have
                some deniability as to what their original data was
              </p>
              <p className="text-lg mt-4">Try for yourself!</p>
              <div className="mt-6 flex flex-col gap-4">
                <div className="w-full">
                  <input
                    type="range"
                    id="noise-level"
                    min="0"
                    max="0.5"
                    step="0.01"
                    value={noiseLevel}
                    onChange={handleSliderChange}
                    className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                  />
                </div>
                <div className="w-full text-base font-medium text-primary-gray">
                  Probability: {noiseLevel.toFixed(2)}
                </div>
              </div>
              <div className="mt-4">
                <button
                  onMouseDown={() => setShowOriginal(true)}
                  onMouseUp={() => setShowOriginal(false)}
                  onMouseLeave={() => setShowOriginal(false)}
                  onTouchStart={() => setShowOriginal(true)}
                  onTouchEnd={() => setShowOriginal(false)}
                  className="px-6 py-2 bg-primary-gray text-primary-white rounded-lg hover:bg-accent transition-colors"
                >
                  Show Original
                </button>
              </div>
            </div>
          </section>

          {/* DP Technique Section */}
          <section className="min-h-[80vh] flex flex-col justify-center p-6">
            <div className="prose prose-lg max-w-none bg-primary-white/80 rounded-lg p-6 shadow-md">
              <p className="text-lg">
                What you see is a Differential Privacy technique called
                Randomized Response. Individual privacy is protected while
                maintaining some statistical patterns of the whole dataset.
                It&apos;s a simple method that satisfies the{" "}
                <span className="font-bold text-accent">
                  Definition of Differential Privacy
                </span>
              </p>
            </div>
          </section>

          {/* Definition Section */}
          <section className="min-h-[80vh] flex flex-col justify-center p-6 mb-8">
            <div className="prose prose-lg max-w-none bg-primary-white/80 rounded-lg p-6 shadow-md">
              <p className="text-lg">
                <span className="font-bold text-accent">
                  Definition of Differential Privacy
                </span>{" "}
                states that for any two datasets that differ in exactly one
                record, the probability of getting the same output from an
                private algorithm should be similar. This paves the way for a{" "}
                <button
                  className="text-accent hover:text-primary-gray cursor-pointer transition-colors inline-flex items-center gap-1 group"
                  onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                >
                  <span className="underline decoration-dotted">
                    mathematical measure of privacy
                  </span>
                  <span
                    className={`inline-block transition-transform duration-200 ${
                      showTechnicalDetails ? "rotate-180" : ""
                    }`}
                  >
                    ▼
                  </span>{" "}
                </button>{" "}
                and a way to quantify the privacy of an algorithm.
              </p>
              <div
                className={`mt-4 transition-all duration-500 overflow-hidden ${
                  showTechnicalDetails
                    ? "max-h-[2000px] opacity-100"
                    : "max-h-0 opacity-0"
                }`}
              >
                <div className="bg-primary-white/90 p-3 md:p-4 rounded-lg border border-primary-gray/10 text-sm md:text-base">
                  <h3 className="text-base font-semibold mb-3 md:mb-4">
                    A more formal definition of Differential Privacy
                  </h3>
                  <p className="mb-3 md:mb-4">
                    A randomized algorithm{" "}
                    <span>
                      <InlineMath math="\mathcal{M}" />
                    </span>{" "}
                    is{" "}
                    <span>
                      <InlineMath math="\varepsilon" />
                    </span>
                    -differentially private if for all pairs of adjacent
                    datasets{" "}
                    <span>
                      <InlineMath math="\mathcal{D}" />
                    </span>{" "}
                    and{" "}
                    <span>
                      <InlineMath math="\mathcal{D}'" />
                    </span>
                    , and for all sets of possible outputs{" "}
                    <span>
                      <InlineMath math="S \subseteq Range(\mathcal{M})" />
                    </span>
                    :
                  </p>
                  <div className="flex justify-center py-2 md:py-4 overflow-x-auto">
                    <div className="scale-[0.6] sm:scale-[0.8] md:scale-100 transform-origin-center min-w-full">
                      <BlockMath math="Pr[\mathcal{M}(\mathcal{D}) \in S] \leq e^{\varepsilon} \cdot Pr[\mathcal{M}(\mathcal{D}') \in S]" />
                    </div>
                  </div>
                  <p className="mb-3 md:mb-4">Where:</p>
                  <ul className="list-disc pl-5 md:pl-6 space-y-1 md:space-y-2 text-xs md:text-sm">
                    <li>
                      <span>
                        <InlineMath math="\varepsilon" />
                      </span>{" "}
                      (epsilon) is the privacy parameter, with smaller values
                      providing stronger privacy
                    </li>
                    <li>
                      <span>
                        <InlineMath math="\mathcal{M}" />
                      </span>{" "}
                      is a randomized algorithm (mechanism)
                    </li>
                    <li>
                      <span>
                        <InlineMath math="\mathcal{D}" />
                      </span>{" "}
                      and{" "}
                      <span>
                        <InlineMath math="\mathcal{D}'" />
                      </span>{" "}
                      are neighbors if they differ by at most one record
                    </li>
                  </ul>
                  <div className="mt-4 space-y-4">
                    <div className="flex flex-col items-center gap-2">
                      <span>Try changing epsilon!</span>
                      <div className="flex flex-wrap justify-between items-center mx-2 gap-2 mt-2">
                        {[0.5, 1, 2, 10].map((epsilon) => (
                          <span
                            key={epsilon}
                            onClick={() => handleEpsilonClick(epsilon)}
                            className={`cursor-pointer ${
                              selectedEpsilon === epsilon
                                ? "text-accent font-bold"
                                : "text-primary-gray hover:text-accent"
                            } transition-colors`}
                          >
                            <span>
                              <InlineMath math={`\\varepsilon=${epsilon}`} />
                            </span>
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <p className="mt-4 text-sm">
                    Read more:{" "}
                    <Link
                      href="https://en.wikipedia.org/wiki/Differential_privacy"
                      target="_blank"
                      className="text-accent hover:text-primary-gray cursor-pointer transition-colors underline decoration-dotted"
                    >
                      Wikipedia
                    </Link>
                  </p>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Visual spacer to ensure proper detection of leaving the SplitView section */}
        <div className="h-1 w-full"></div>
      </div>

      {/* Desktop layout - Horizontal split */}
      <div className="hidden md:flex">
        {/* Left side - content */}
        <div className="w-1/2 order-1">
          {/* Title Section */}
          <section
            ref={titleRef as React.LegacyRef<HTMLElement>}
            className="h-screen snap-start flex flex-col justify-center p-12"
          >
            <Hero
              titleRef={titleRef}
              scrollToSection={scrollToSection}
              whatIsDPRef={SplitViewRef}
              methodsRef={methodsRef}
              discussionRef={discussionRef}
            />
          </section>

          {/* Keep the existing desktop sections unchanged */}
          {/* First Paragraph */}
          <section
            ref={SplitViewRef as React.LegacyRef<HTMLElement>}
            className="h-screen snap-start flex flex-col justify-center p-12"
          >
            <div className="prose prose-lg max-w-none">
              <h1 className="text-4xl font-bold mb-4">
                What is Differential Privacy?
              </h1>
              <p className="text-xl">
                In today&apos;s data-driven world, the need to protect
                individual privacy while maintaining the utility of data
                analysis has become increasingly crucial. Differential Privacy
                (DP) emerges as a mathematical framework that provides strong
                privacy guarantees while allowing meaningful statistical
                analysis
              </p>
            </div>
          </section>

          {/* Second Paragraph */}
          <section className="h-screen snap-start flex flex-col justify-center p-12">
            <div className="prose prose-lg max-w-none">
              <p className="text-xl">
                At its core, differential privacy ensures that the presence or
                absence of any individual&apos;s data in a dataset does not
                significantly affect the results of any analysis performed on
                that dataset. This is achieved by carefully introducing random
                noise into the computation process, making it virtually
                impossible to reverse-engineer individual records...
              </p>
            </div>
          </section>

          {/* Third Paragraph */}
          <section
            ref={addNoiseML}
            className="h-screen snap-start flex flex-col justify-center p-12"
          >
            <div className="prose prose-lg max-w-none">
              <p className="text-xl">
                ...all while maintaining the big picture!
              </p>
              <p className="text-xl mt-6">
                But what just happened? We added noise to the image of Mona Lisa
                by probabilistically flipping each pixel. This way, you can
                still see the big picture, but the individual pixels&apos; have
                some deniability as to what their original data was
              </p>
              <p className="text-xl mt-6">Try for yourself!</p>
              <div className="mt-8 flex items-center gap-4">
                <div className="w-1/2">
                  <input
                    type="range"
                    id="noise-level"
                    min="0"
                    max="0.5"
                    step="0.01"
                    value={noiseLevel}
                    onChange={handleSliderChange}
                    className="w-full h-2 bg-primary-gray rounded-lg appearance-none cursor-pointer accent-accent"
                  />
                </div>
                <div className="w-1/2 text-lg font-medium text-primary-gray">
                  Probability: {noiseLevel.toFixed(2)}
                </div>
              </div>
              <div className="mt-4 flex">
                <button
                  onMouseDown={() => setShowOriginal(true)}
                  onMouseUp={() => setShowOriginal(false)}
                  onMouseLeave={() => setShowOriginal(false)}
                  className="px-6 py-2 bg-primary-gray text-primary-white rounded-lg hover:bg-accent transition-colors"
                >
                  Show Original
                </button>
              </div>
            </div>
          </section>

          {/* Fourth Paragraph */}
          <section className="h-screen snap-start flex flex-col justify-center p-12">
            <div className="prose prose-lg max-w-none">
              <p className="text-xl">
                What you see is a Differential Privacy technique called
                Randomized Response. Individual privacy is protected while
                maintaining some statistical patterns of the whole dataset.
                It&apos;s a simple method that satisfies the{" "}
                <span className="font-bold text-accent">
                  Definition of Differential Privacy
                </span>
              </p>
            </div>
          </section>

          {/* Fifth Paragraph */}
          <section className="h-screen snap-start flex flex-col justify-center p-12">
            <div className="prose prose-lg max-w-none">
              <p className="text-xl">
                <span className="font-bold text-accent">
                  Definition of Differential Privacy
                </span>{" "}
                states that for any two datasets that differ in exactly one
                record, the probability of getting the same output from an
                private algorithm should be similar. This paves the way for a{" "}
                <button
                  className="text-accent hover:text-primary-gray cursor-pointer transition-colors inline-flex items-center gap-1 group"
                  onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                >
                  <span className="underline decoration-dotted">
                    mathematical measure of privacy
                  </span>
                  <span
                    className={`inline-block transition-transform duration-200 ${
                      showTechnicalDetails ? "rotate-180" : ""
                    }`}
                  >
                    ▼
                  </span>{" "}
                </button>{" "}
                and a way to quantify the privacy of an algorithm.
              </p>
              <div
                className={`mt-4 transition-all duration-500 overflow-hidden ${
                  showTechnicalDetails
                    ? "max-h-[500px] opacity-100"
                    : "max-h-0 opacity-0"
                }`}
              >
                <div className="bg-primary-gray/5 p-6 rounded-lg border border-primary-gray/10">
                  <h3 className="text-lg font-semibold mb-4">
                    A more formal definition of Differential Privacy
                  </h3>
                  <p className="mb-4">
                    A randomized algorithm{" "}
                    <span>
                      <InlineMath math="\mathcal{M}" />
                    </span>{" "}
                    is{" "}
                    <span>
                      <InlineMath math="\varepsilon" />
                    </span>
                    -differentially private if for all pairs of adjacent
                    datasets{" "}
                    <span>
                      <InlineMath math="\mathcal{D}" />
                    </span>{" "}
                    and{" "}
                    <span>
                      <InlineMath math="\mathcal{D}'" />
                    </span>
                    , and for all sets of possible outputs{" "}
                    <span>
                      <InlineMath math="S \subseteq Range(\mathcal{M})" />
                    </span>
                    :
                  </p>
                  <div className="flex justify-center py-4 overflow-x-auto">
                    <div className="scale-[0.6] sm:scale-[0.8] md:scale-100 transform-origin-center min-w-full">
                      <BlockMath math="Pr[\mathcal{M}(\mathcal{D}) \in S] \leq e^{\varepsilon} \cdot Pr[\mathcal{M}(\mathcal{D}') \in S]" />
                    </div>
                  </div>
                  <p className="mb-4">Where:</p>
                  <ul className="list-disc pl-6 space-y-2 text-sm md:text-base">
                    <li>
                      <span>
                        <InlineMath math="\varepsilon" />
                      </span>{" "}
                      (epsilon) is the privacy parameter, with smaller values
                      providing stronger privacy
                    </li>
                    <li>
                      <span>
                        <InlineMath math="\mathcal{M}" />
                      </span>{" "}
                      is a randomized algorithm (mechanism)
                    </li>
                    <li>
                      <span>
                        <InlineMath math="\mathcal{D}" />
                      </span>{" "}
                      and{" "}
                      <span>
                        <InlineMath math="\mathcal{D}'" />
                      </span>{" "}
                      are neighbors if they differ by at most one record
                    </li>
                  </ul>
                  <div className="mt-4 space-y-4">
                    <div className="flex items-center gap-2">
                      <span>Try changing epsilon!</span>
                      <div className="flex-1 flex justify-between items-center mx-4">
                        {[0.5, 1, 2, 10].map((epsilon) => (
                          <span
                            key={epsilon}
                            onClick={() => handleEpsilonClick(epsilon)}
                            className={`cursor-pointer ${
                              selectedEpsilon === epsilon
                                ? "text-accent font-bold"
                                : "text-primary-gray hover:text-accent"
                            } transition-colors`}
                          >
                            <span>
                              <InlineMath math={`\\varepsilon=${epsilon}`} />
                            </span>
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <p className="mt-4">
                    Read more:{" "}
                    <Link
                      href="https://en.wikipedia.org/wiki/Differential_privacy"
                      target="_blank"
                      className="text-accent hover:text-primary-gray cursor-pointer transition-colors underline decoration-dotted"
                    >
                      Wikipedia
                    </Link>
                  </p>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Right side - Mona Lisa */}
        <div className="w-1/2 sticky top-0 h-screen bg-primary-white flex items-center justify-center overflow-hidden order-2">
          <div className="transform scale-125">
            <MonaLisa
              ascii={monaLisa}
              shouldAddNoise={pastNoiseTrigger}
              noiseLevel={noiseLevel}
              showOriginal={showOriginal}
            />
          </div>
        </div>
      </div>
    </>
  );
}
