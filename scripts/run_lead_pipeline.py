import argparse
import sys
import logging
from server.lead_pipeline.core.csv_manager import UnifiedCSVManager
from server.lead_pipeline.core.compliance import ComplianceGate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Kalpvruksh Finserv Lead Pipeline Runner")
    parser.add_argument("--source", type=str, required=True, 
                        choices=["ddg", "nominatim", "gmaps", "bizkonnect", "linkedin", "practo", "justdial"],
                        help="The lead source collector to use")
    parser.add_argument("--query", type=str, help="Search query (e.g., 'doctors in pune')")
    parser.add_argument("--file", type=str, help="Path to input file (for bizkonnect, linkedin, practo, justdial)")
    parser.add_argument("--limit", type=int, default=20, help="Maximum leads to fetch (for live search APIs)")
    
    args = parser.parse_args()
    
    csv_manager = UnifiedCSVManager()
    compliance_gate = ComplianceGate()
    leads = []
    
    if args.source == "ddg":
        if not args.query:
            logger.error("--query is required for DDG source")
            sys.exit(1)
        from server.lead_pipeline.collectors.collector_ddg import DDGCollector
        leads = DDGCollector().fetch_real_leads(args.query, args.limit)
        
    elif args.source == "nominatim":
        if not args.query:
            logger.error("--query is required for Nominatim source")
            sys.exit(1)
        from server.lead_pipeline.collectors.collector_nominatim import NominatimCollector
        leads = NominatimCollector().fetch_real_leads(args.query, args.limit)
        
    elif args.source == "gmaps":
        if not args.query:
            logger.error("--query is required for Google Maps source")
            sys.exit(1)
        from server.lead_pipeline.collectors.collector_gmaps import GoogleMapsCollector
        leads = GoogleMapsCollector().fetch_business_leads(args.query, "Pune", args.limit)
        
    elif args.source == "bizkonnect":
        if not args.file:
            logger.error("--file is required for BizKonnect importer")
            sys.exit(1)
        from server.lead_pipeline.collectors.import_bizkonnect import BizKonnectImporter
        leads = BizKonnectImporter().import_csv(args.file)
        
    elif args.source == "linkedin":
        if not args.file:
            logger.error("--file is required for LinkedIn importer")
            sys.exit(1)
        from server.lead_pipeline.collectors.linkedin_discovery_import import LinkedInDiscoveryImporter
        leads = LinkedInDiscoveryImporter().import_discovery_csv(args.file)
        
    elif args.source == "practo":
        if not args.file:
            logger.error("--file is required for Practo collector")
            sys.exit(1)
        from server.lead_pipeline.collectors.collector_practo import PractoAssistedCollector
        leads = PractoAssistedCollector().extract_from_html(args.file)
        
    elif args.source == "justdial":
        if not args.file:
            logger.error("--file is required for JustDial collector")
            sys.exit(1)
        from server.lead_pipeline.collectors.collector_justdial import JustDialAssistedCollector
        leads = JustDialAssistedCollector().extract_from_html(args.file)

    if not leads:
        logger.info("No leads generated/imported.")
        return

    # Filter compliant leads
    compliant_leads = []
    for lead in leads:
        if compliance_gate.is_lead_callable(lead):
            compliant_leads.append(lead)
        else:
            logger.info(f"Lead {lead.name} blocked by compliance gate.")
            
    if compliant_leads:
        csv_manager.save_leads(compliant_leads)
        logger.info(f"Pipeline finished. {len(compliant_leads)} compliant leads saved.")
    else:
        logger.info("Pipeline finished. No compliant leads to save.")

if __name__ == "__main__":
    main()
