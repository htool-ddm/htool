// #include <htool/misc/configuration.hpp>
#include <array>
#include <htool/misc/logger.hpp>
#include <sstream>

class TestWriter : public htool::IObjectWriter {
  public:
    std::string m_string = "";
    htool::LogLevel m_current_log_level;
    void write(htool::LogLevel log_level, const std::string &message) override {
        if (log_level <= m_current_log_level) {
            m_string.append(logging_level_to_string(log_level) + message);
        }
    }
    void set_log_level(htool::LogLevel log_level) override { m_current_log_level = log_level; }
    htool::LogLevel get_log_level() { return m_current_log_level; }
};

int main() {
    std::array<htool::LogLevel, 5> log_levels{
        htool::LogLevel::CRITICAL,
        htool::LogLevel::ERROR,
        htool::LogLevel::WARNING,
        htool::LogLevel::DEBUG,
        htool::LogLevel::INFO};

    auto test_writer = std::make_shared<TestWriter>();
    for (auto log_level : log_levels) {
        for (auto verbosity : {0, 5, 10, 15, 20, 25, 30, 40}) {
            auto &logger = htool::Logger::get_instance();
            logger.set_current_log_level(verbosity);
            logger.set_current_writer(test_writer);
            unsigned int current_log_level = static_cast<unsigned int>(test_writer->get_log_level());
            logger.log(log_level, "current log level: " + std::to_string(current_log_level) + "\n");
        }
    }

    std::string ref_string = R"([Htool critical] current log level: 0
[Htool critical] current log level: 5
[Htool critical] current log level: 10
[Htool critical] current log level: 15
[Htool critical] current log level: 20
[Htool critical] current log level: 25
[Htool critical] current log level: 30
[Htool critical] current log level: 40
[Htool error]    current log level: 10
[Htool error]    current log level: 15
[Htool error]    current log level: 20
[Htool error]    current log level: 25
[Htool error]    current log level: 30
[Htool error]    current log level: 40
[Htool warning]  current log level: 20
[Htool warning]  current log level: 25
[Htool warning]  current log level: 30
[Htool warning]  current log level: 40
[Htool debug]    current log level: 30
[Htool debug]    current log level: 40
[Htool info]     current log level: 40
)";
    if (!(ref_string == test_writer->m_string)) {
        std::cout << test_writer->m_string;
    }
    return !(ref_string == test_writer->m_string);
}
