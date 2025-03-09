import { XMarkIcon } from "@heroicons/react/24/outline";
import Image from "next/image";
import Link from "next/link";
import contactsData from "@/data/contacts.json";
import { getPublicPath } from "@/lib/utils";

interface ContactModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ContactModal({ isOpen, onClose }: ContactModalProps) {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center"
      onClick={onClose}
    >
      <div 
        className="bg-white rounded-lg p-8 max-w-4xl w-full mx-4 relative"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-primary-gray hover:text-accent transition-colors"
        >
          <XMarkIcon className="w-6 h-6" />
        </button>

        <h2 className="text-2xl font-bold mb-8 text-center">Contact Us</h2>

        {/* 2x2 Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {contactsData.contacts.map((contact) => (
            <div key={contact.name} className="flex items-start space-x-4">
              {/* Profile Picture */}
              <div className="w-24 h-24 rounded-full overflow-hidden flex-shrink-0">
                <Image
                  src={getPublicPath(contact.image)}
                  alt={`${contact.name}'s profile`}
                  width={96}
                  height={96}
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Contact Information */}
              <div className="flex-1">
                <h3 className="font-bold text-lg mb-2">{contact.name}</h3>
                <div className="space-y-1">
                  <p className="text-primary-gray">{contact.email}</p>
                  <div className="flex flex-wrap gap-2">
                    <Link
                      href={contact.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-accent hover:text-primary-gray transition-colors"
                    >
                      GitHub
                    </Link>
                    <Link
                      href={contact.linkedin}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-accent hover:text-primary-gray transition-colors"
                    >
                      LinkedIn
                    </Link>
                    {contact.website && (
                      <Link
                        href={contact.website}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-accent hover:text-primary-gray transition-colors"
                      >
                        Website
                      </Link>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 