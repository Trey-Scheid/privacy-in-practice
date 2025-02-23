import { PaperDisplay } from "./paper-display/PaperDisplay";

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
            FILLER TEXT Telemetry data is collected from the user&apos;s device
            and sent to the server. We apply DP to the telemetry data to protect
            the user&apos;s privacy. FILLER TEXT Telemetry data is collected
            from the user&apos;s device and sent to the server. We apply DP to
            the telemetry data to protect the user&apos;s privacy. FILLER TEXT
            Telemetry data is collected from the user&apos;s device and sent to
            the server. We apply DP to the telemetry data to protect the
            user&apos;s privacy.
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
            <p className="text-xl text-primary-gray">
              Something about telemetry here. Lorem ipsum dolor sit amet,
              consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut
              labore et dolore magna aliqua. Ut enim ad minim veniam, quis
              nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
              consequat.
            </p>
          </div>
        </section>

        {/* Second Section */}
        <section className="mb-16">
          <div className="prose prose-lg max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold mb-4 text-primary-gray">
              Applying DP to Telemetry Data
            </h2>
            <p className="text-xl text-primary-gray">
              Additional content can go here and will scroll normally.
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
