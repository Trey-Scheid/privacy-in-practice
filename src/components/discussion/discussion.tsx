interface DiscussionProps {
  discussionRef: React.RefObject<HTMLElement>;
}

function Discussion({ discussionRef }: DiscussionProps) {
  return (
    <div className="bg-primary-gray text-primary-white">
      {/* Title Section */}
      <section
        ref={discussionRef as React.LegacyRef<HTMLElement>}
        className="flex flex-col p-12"
      >
        <div className="prose prose-lg max-w-3xl mx-auto mt-[20vh]">
          <h1 className="text-4xl font-bold mb-4">
            Discussion
          </h1>
          <p className="text-xl">
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
            <h2 className="text-3xl font-bold mb-4">
              Randomized Response
            </h2>
            <p className="text-xl">
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
            <h2 className="text-3xl font-bold mb-4">
              Laplace Mechanism
            </h2>
            <p className="text-xl">
              Additional content can go here and will scroll normally.
            </p>
          </div>
        </section>

        <section className="mb-16">
          <div className="prose prose-lg max-w-3xl mx-auto">
                <h2 className="text-3xl font-bold mb-4">
              Meta-Analysis
            </h2>
            <p className="text-xl">
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

export default Discussion;
