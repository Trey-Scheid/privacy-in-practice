interface DiscussionProps {
  discussionRef: React.RefObject<HTMLElement>;
}

function Discussion({ discussionRef }: DiscussionProps) {
  return (
    <div className="bg-primary-gray text-primary-white relative z-20">
      {/* Title Section */}
      <section
        ref={discussionRef as React.LegacyRef<HTMLElement>}
        className="flex flex-col p-4 md:p-12 pt-20 md:pt-12"
      >
        <div className="prose prose-lg max-w-3xl mx-auto mt-8 md:mt-[20vh] space-y-4 md:space-y-8">
          <h1 className="text-2xl md:text-4xl font-bold mb-2 md:mb-4">
            The Feasibility of Applying DP
          </h1>
          <p className="text-base md:text-xl">
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
      <div className="py-6 md:py-12">
        <div className="max-w-3xl mx-auto space-y-8 md:space-y-16">
          {/* First Section - Randomized Response */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Header 1
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl">
                  Something about telemetry here. Lorem ipsum dolor sit amet,
                  consectetur adipiscing elit. Sed do eiusmod tempor incididunt
                  ut labore et dolore magna aliqua. Ut enim ad minim veniam,
                  quis nostrud exercitation ullamco laboris nisi ut aliquip ex
                  ea commodo consequat.
                </p>
              </div>
            </div>
          </section>

          {/* Second Section */}
          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Header 2
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl">
                  Additional content can go here and will scroll normally.
                </p>
              </div>
            </div>
          </section>

          <section>
            <div className="prose prose-lg max-w-none px-4 md:px-0">
              <h2 className="text-xl md:text-3xl font-bold mb-2 md:mb-4">
                Header 3
              </h2>
              <div className="space-y-4 md:space-y-8">
                <p className="text-base md:text-xl">
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
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default Discussion;
