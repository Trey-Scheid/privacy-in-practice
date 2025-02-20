interface OurMethodsProps {
  methodsRef: React.RefObject<HTMLElement | null>;
}

function OurMethods({ methodsRef }: OurMethodsProps) {
  return (
    <div className="bg-accent-light" >
      <div className="flex">
        {/* Left side - content */}
        <div className="w-full">
          {/* Title Section */}
          <section
            ref={methodsRef}
            className="h-screen snap-start flex flex-col justify-center p-12"
          >
            <div className="prose prose-lg max-w-3xl mx-auto">
              <h1 className="text-4xl font-bold mb-4 text-primary-gray">
                How We Applied DP
              </h1>
              <p className="text-xl text-primary-gray">
                (Incomplete) We explore various techniques in differential privacy, from simple
                randomization to complex mechanisms. Each method offers unique
                trade-offs between privacy and utility.
              </p>
            </div>
          </section>

          {/* First Section - Randomized Response */}
          <section
            className="h-screen snap-start flex flex-col justify-center p-12"
          >
            <div className="prose prose-lg max-w-3xl mx-auto">
              <p className="text-xl text-primary-gray">
                (Incomplete) What&apos;s important to note is that Differential Privacy is a
                property of an algorithm, not a property of the data. Another
                might be to anonymize data by removing personally identifiable
                information, but this isn&apos;t enough to guarantee privacy on its
                own.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default OurMethods;