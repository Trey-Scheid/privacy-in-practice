import { PaperDisplay } from "./paper-display/PaperDisplay";
import { InlineMath } from "react-katex";

interface OurMethodsProps {
  methodsRef: React.RefObject<HTMLElement>;
}

function OurMethods({ methodsRef }: OurMethodsProps) {
  return (
    <div className="bg-accent-light">
      {/* Title Section */}
      <section
        ref={methodsRef as React.LegacyRef<HTMLElement>}
        className="flex flex-col p-12"
      >
        <div className="prose prose-lg max-w-3xl mx-auto mt-[20vh]">
          <h1 className="text-4xl font-bold mb-4 text-primary-gray">
            How We Applied DP
          </h1>
          <p className="text-xl text-primary-gray">
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
      <div className="py-12">
        {/* First Section - Randomized Response */}
        <section className="mb-16">
          <div className="prose prose-lg max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              What is Telemetry Data?
            </h2>
            <p className="text-xl text-primary-gray mb-4">
              So why would we want to use DP on telemetry data? Telemetry data
              is data collected by devices such as CPUs or hard drives to
              monitor things like temperature, power usage, crash metrics, and
              much, much more. It is often collected in large volumes and is
              used for diagnostic analysis to improve user experiences.
            </p>
            <p className="text-xl text-primary-gray mb-4">
              Each piece of data is typically attributed to a specific device
              ID, which if somebody could link these to a specific user, they
              could know a lot about that user based on how they use their
              device. High use of their GPU from noon to 1 PM? Maybe
              they&apos;re a video editor or they&apos;re playing a video game.
              Maybe you know a bunch of devices that are used as AWS servers.
              Knowing how they operate could be valuable to a competitor.
            </p>
            <p className="text-xl text-primary-gray">
              This is why we want to apply DP to telemetry data. We want to
              collect data in a way that allows us to perform useful analyses
              without compromising user privacy.
            </p>
          </div>
        </section>

        {/* Second Section */}
        <section className="mb-12">
          <div className="prose prose-lg max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              Applying DP to Telemetry Data
            </h2>
            <p className="text-xl text-primary-gray mb-4">
              For our study, we looked at four different papers that used a
              variety of different methods in their analyses. We first recreated
              the study non-privately with the same volume of data to serve as a
              baseline. Then, based on the methods used in each paper, we
              applied differential privacy at a variety of different privacy
              levels to assess how different the final analysis results were
              from our baseline.
            </p>
            <p className="text-xl text-primary-gray">
              Below you can look at the results of each paper.
            </p>
          </div>
        </section>

        <PaperDisplay />

        <section className="mb-16">
          <div className="prose prose-lg max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              Meta-Analysis
            </h2>
            <p className="text-xl text-primary-gray">
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
              Additional content can go here and will scroll normally.
            </p>
          </div>
        </section>

        <section className="mb-16">
          <div className="prose prose-lg max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              Conclusion
            </h2>
            <p className="text-xl text-primary-gray">
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
              Additional content can go here and will scroll normally.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

export default OurMethods;
