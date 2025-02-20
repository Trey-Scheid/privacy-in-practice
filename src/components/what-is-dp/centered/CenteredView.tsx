import Link from "next/link";
import Image from "next/image";
import { HistogramViz } from "./histogram/HistogramViz";
import { ConfidenceViz } from "./confidence/ConfidenceViz";
export function CenteredView() {
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
                src="/netflix.svg"
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
            because they don&apos;t want to reveal if someone was in a training
            set. The question then becomes, where can we effectively add noise
            in order to preserve privacy but still get an accurate model?
          </p>
        </div>
      </section>

      <section className="h-screen snap-start flex flex-col justify-center p-12">
        <ConfidenceViz />
      </section>
    </div>
  );
}
