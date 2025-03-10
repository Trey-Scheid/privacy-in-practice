import { PaperDisplay } from "./paper-display/PaperDisplay";
import { InlineMath } from "react-katex";
import Image from "next/image";

interface OurMethodsProps {
  methodsRef: React.RefObject<HTMLElement>;
}

function OurMethods({ methodsRef }: OurMethodsProps) {
  return (
    <div className="bg-accent-light">
      {/* Title Section */}
      <section
        ref={methodsRef as React.LegacyRef<HTMLElement>}
        className="flex flex-col p-4 md:p-12 pt-20 md:pt-12"
      >
        <div className="prose prose-lg max-w-3xl mx-auto mt-8 md:mt-[20vh] space-y-4 md:space-y-8">
          <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-4 text-primary-gray">
            How We Applied DP
          </h1>
          <p className="text-base md:text-xl text-primary-gray">
            We sought to assess how feasible it is to apply DP to real-world
            data analysis tasks. Adding noise often leads to reduced utility, so
            we focused on recreating four different papers that did not use DP
            and compared their results to our DP-applied counterparts. Our focus
            was on tasks using Intel telemetry data to create realistic,
            high-data volume analyses. We found that for high-privacy settings{" "}
            <span>
              (<InlineMath math="\varepsilon \leq 1" />)
            </span>{" "}
            the utility loss is often too great to be practical.
          </p>
        </div>
      </section>

      {/* Content sections */}
      <div className="py-6 md:py-12">
        {/* First Section - Telemetry Data */}
        <div className="max-w-3xl mx-auto space-y-8 md:space-y-16 px-4 md:px-0">
          <section>
            <div className="prose prose-lg max-w-none">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4 text-primary-gray">
                What is Telemetry Data?
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl text-primary-gray">
                  So why would we want to use DP on telemetry data? Telemetry
                  data is data collected by devices such as CPUs or hard drives
                  to monitor things like temperature, power usage, crash
                  metrics, and much, much more. It is often collected in large
                  volumes and is used for diagnostic analysis to improve user
                  experiences.
                </p>
                <p className="text-base md:text-xl text-primary-gray">
                  Each piece of data is typically attributed to a specific
                  device ID, which if somebody could link these to a specific
                  user, they could know a lot about that user based on how they
                  use their device. High use of their GPU from noon to 1 PM?
                  Maybe they&apos;re a video editor or they&apos;re playing a
                  video game. Maybe you know a bunch of devices that are used as
                  AWS servers. Knowing how they operate could be valuable to a
                  competitor.
                </p>
                <p className="text-base md:text-xl text-primary-gray">
                  This is why we want to apply DP to telemetry data. We want to
                  collect data in a way that allows us to perform useful
                  analyses without compromising user privacy.
                </p>
              </div>
            </div>
          </section>

          {/* Second Section */}
          <section>
            <div className="prose prose-lg max-w-none">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4 text-primary-gray">
                Applying DP to Telemetry Data
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl text-primary-gray">
                  For our study, we looked at four different papers that used a
                  variety of different methods in their analyses. We first
                  recreated the study non-privately with the same volume of data
                  to serve as a baseline. Then, based on the methods used in
                  each paper, we applied differential privacy at a variety of
                  different privacy levels to assess how different the final
                  analysis results were from our baseline.
                </p>
                <p className="text-base md:text-xl text-primary-gray">
                  Below you can look at the results of each paper.
                </p>
              </div>
            </div>
          </section>
        </div>

        {/* PaperDisplay - Full Width */}
        <div className="w-full">
          <PaperDisplay />
        </div>

        {/* Remaining sections */}
        <div className="max-w-3xl mx-auto space-y-8 md:space-y-16 px-4 md:px-0">
          <section>
            <div className="prose prose-lg max-w-none">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4 text-primary-gray">
                Meta-Analysis
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl text-primary-gray">
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                  Additional content can go here and will scroll normally.
                </p>

                {/* Image container with responsive width */}
                <div className="flex justify-center w-full my-6 md:my-8">
                  <div className="w-2/3 relative">
                    <Image
                      src={"meta.png"}
                      alt="Meta-analysis visualization"
                      width={800}
                      height={500}
                      className="w-full h-auto rounded-lg shadow-md"
                      priority
                    />
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default OurMethods;
