import Link from "next/link";
import Image from "next/image";
import { HistogramViz } from "./histogram/HistogramViz";
import { ConfidenceViz } from "./confidence/ConfidenceViz";
import { getPublicPath } from "@/lib/utils";
import { useState } from "react";

export function CenteredView() {
  const [showRevealDetails, setShowRevealDetails] = useState(false);

  return (
    <div className="bg-primary-white">
      {/* Sixth Paragraph - Full Width Centered */}
      <section className="h-screen snap-start flex flex-col justify-center items-center p-12">
        <div className="prose prose-lg max-w-3xl mx-auto text-center">
          <p className="text-xl">
            What&apos;s important to note is that Differential Privacy is a
            property of an algorithm, not a property of the data. Another
            intuitive{" "}
            <Link
              href="https://en.wikipedia.org/wiki/Privacy-enhancing_technologies"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
            >
              Privacy-Enhancing Technology
            </Link>{" "}
            might be to anonymize data by removing personally identifiable
            information, but this isn&apos;t enough to guarantee privacy on its
            own.
          </p>
        </div>
      </section>

      {/* Netflix Prize Section */}
      <section className="h-screen snap-start flex justify-between items-center p-12 mx-12">
        <div className="w-1/2 prose prose-lg max-w-none">
          <p className="text-xl">
            In 2006, Netflix created the{" "}
            <Link
              href="https://en.wikipedia.org/wiki/Netflix_Prize"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
            >
              Netflix Prize
            </Link>
            , a competition to improve Netflix&apos;s movie recommendation
            algorithm. The dataset used in the competition contained 100 million
            ratings from 480,000 users on 17,770 movies, anonymized by removing
            personally identifiable information. One year later, using iMDB
            ratings as a reference, two researchers from UT Austin were able to{" "}
            <Link
              href="https://arxiv.org/abs/cs/0610105"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
            >
              deanonymize the 99% of the users in the dataset
            </Link>
            . This is why the rigor of DP is so important.
          </p>
        </div>
        <div className="w-1/2 flex justify-center items-center">
          <div className="w-full h-full flex justify-center items-center">
            <div className="relative w-[70%]">
              <Image
                src={getPublicPath("netflix.svg")}
                alt="Netflix Logo"
                className="w-full h-auto"
                width={0}
                height={0}
                sizes="(max-width: 768px) 100vw, 50vw"
                style={{ width: "100%", height: "auto" }}
                priority
              />
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Visualization Section */}
      <section className="h-screen snap-start flex flex-col justify-center p-12">
        <HistogramViz />
      </section>

      {/* Logit Viz Section */}
      <section className="h-screen snap-start flex flex-col justify-center items-center p-12">
        <div className="prose prose-lg max-w-3xl mx-auto text-center">
          <p className="text-xl">
            Differential Privacy can be used for more complex queries too, such
            as training a machine learning model. A data scientist may do this
            because they don&apos;t want to{" "}
            <button
              className="text-accent hover:text-primary-gray cursor-pointer transition-colors inline-flex items-center gap-1 group"
              onClick={() => setShowRevealDetails(!showRevealDetails)}
            >
              <span className="underline decoration-dotted">
                reveal if someone was in a training set
              </span>
              <span
                className={`inline-block transition-transform duration-200 ${
                  showRevealDetails ? "rotate-180" : ""
                }`}
              >
                ▼
              </span>
            </button>
            {". "}
          </p>
          <p className="text-xl">The question then becomes:</p>
          <p className="text-xl font-bold text-accent mt-4">
            When are differentially private methods practical and useful?
          </p>
          <p className="text-xl font-bold text-accent">
            How effective is differential privacy when applied in practice?
          </p>
          <div
            className={`mt-8 transition-all duration-500 overflow-hidden ${
              showRevealDetails
                ? "max-h-[500px] opacity-100"
                : "max-h-0 opacity-0"
            }`}
          >
            <div className="bg-primary-gray/5 p-6 rounded-lg border border-primary-gray/10">
              <h3 className="text-lg font-semibold mb-4">
                Models can reveal what they were trained on!
              </h3>
              <p className="text-lg mb-4">
                When ChatGPT 3.5 was released, trained on many datasets publicly
                and privately available. Clever prompters called "Agents" were
                able to gather SSN's for individuals which the model would
                produce with perfect accuracy! A differentially private
                algorithm guarantees that model outputs are not significantly
                different wether your SSN was in the training data or not, which
                means that the model would not reveal any information about the
                training data.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="h-screen snap-start flex flex-col justify-center p-12">
        <ConfidenceViz />
      </section>

      <section className="h-screen snap-start flex flex-col justify-center items-center p-12">
        <div className="prose prose-lg max-w-3xl mx-auto text-center">
          <p className="text-xl">Conclusion + transition here!!!</p>
        </div>
      </section>
    </div>
  );
}
