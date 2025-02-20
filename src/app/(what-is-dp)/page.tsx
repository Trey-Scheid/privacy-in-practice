"use client";

import { useEffect, useState, useRef } from "react";
import { useInView } from "react-intersection-observer";
import { SplitView } from "../../components/what-is-dp/split/SplitView";
import { NavBar } from "../../components/ui/NavBar";
import { CenteredView } from "../../components/what-is-dp/centered/CenteredView";

export default function Home() {
  const [monaLisa, setMonaLisa] = useState("");
  const { ref: titleRef, inView: isTitleVisible } = useInView({
    threshold: 0.5,
  });

  // Refs for sections
  const whatIsDPRef = useRef<HTMLElement>(null);

  const scrollToSection = (ref: React.RefObject<HTMLElement | null>) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    fetch("/monalisa.txt")
      .then((response) => response.text())
      .then((content) => setMonaLisa(content))
      .catch((error) => {
        console.error("Failed to load Mona Lisa:", error);
        setMonaLisa("");
      });
  }, []);

  return (
    <div className="h-screen overflow-y-scroll snap-y snap-mandatory bg-primary-white text-primary-black">
      {/* Navigation Bar */}
      <NavBar
        isTitleVisible={isTitleVisible}
        scrollToSection={scrollToSection}
        whatIsDPRef={whatIsDPRef}
      />

      {/* What is DP Section */}
      <SplitView
        monaLisa={monaLisa}
        SplitViewRef={whatIsDPRef}
        titleRef={titleRef}
        scrollToSection={scrollToSection}
      />

      {/* Histogram Section */}
      <CenteredView />
    </div>
  );
}
