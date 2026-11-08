#include <gfa_parser.hpp>

#include <fstream>
#include <iostream>

static void require(bool ok, const char* why)
{
    if (!ok)
        throw std::runtime_error(why);
}

static void check_graph(const BiGraph& g)
{
    require(g.n_nodes == 3, "wrong node count");
    require(g.node_names == std::vector<std::string>({"1", "2", "3"}), "wrong names/order");
    require(g.links.size() == 3, "wrong native link count");
    const auto& a = g.links[0];
    const auto& b = g.links[1];
    const auto& c = g.links[2];
    require(a.src == 0 && a.dst == 1 && a.orient_src == '+' && a.orient_dst == '-', "wrong first link");
    require(b.src == 1 && b.dst == 2 && b.orient_src == '-' && b.orient_dst == '+', "wrong second link");
    require(c.src == 2 && c.dst == 0 && c.orient_src == '+' && c.orient_dst == '+', "wrong final link");
}

int main(int argc, char** argv)
{
    try {
        if (argc >= 3 && std::string(argv[1]) == "--dump") {
            const auto g = GFAParser::parse_file(argv[2], argc > 3 ? std::stoi(argv[3]) : 1);
            std::cout << g.n_nodes << '\n';
            for (const auto& name : g.node_names)
                std::cout << name.size() << ':' << name << '\n';
            for (const auto& e : g.links)
                std::cout << e.src << '\t' << e.orient_src << '\t' << e.dst << '\t' << e.orient_dst << '\n';
            return 0;
        }
        char dir[] = "/tmp/bf-gfa-reader-XXXXXX";
        require(mkdtemp(dir) != nullptr, "mkdtemp failed");
        const std::string path = std::string(dir) + "/window.gfa";
        {
            std::ofstream f(path, std::ios::binary);
            f << "H\tVN:Z:1.1\nS\t1\tA\nS\t2\tC\nS\t3\tG\nL\t1\t+\t2\t-\t0M\n";
            // Names must survive the release of earlier mapped pages.
            f << "W\tx\t0\tchr1\t0\t1\t";
            const std::string block(1 << 20, '>');
            for (int i = 0; i < 40; ++i)
                f.write(block.data(), block.size());
            f << "\nL\t2\t-\t3\t+\t0M\nL\t3\t+\t1\t+\t0M";
            require(f.good(), "fixture write failed");
        }
        check_graph(GFAParser::parse_file(path, 1));
        // A second read must start with its own window.
        check_graph(GFAParser::parse_file(path, 4));
        std::remove(path.c_str());
        rmdir(dir);
        std::cout << "GFA reader window, names, orientation and repeated-parse checks PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
